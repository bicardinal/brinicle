#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <queue>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <string>
#include <malloc.h>
#include <stdint.h>

#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

#include "operations.h"
#include <sys/file.h>


#if defined(__GNUC__) || defined(__clang__)
  #define PREFETCH_R(addr, lvl) __builtin_prefetch((addr), 0, (lvl))
#elif defined(_MSC_VER)
  #include <immintrin.h>
  #define PREFETCH_R(addr, lvl) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
  #define PREFETCH_R(addr, lvl) ((void)0)
#endif

#define GHNSW_PREFETCH(addr) PREFETCH_R((addr), 1)


static inline size_t align_up(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

static inline uint8_t* aligned_malloc64(size_t nbytes) {
#if defined(_MSC_VER)
	return reinterpret_cast<uint8_t*>(_aligned_malloc(nbytes, 64));
#else
	size_t padded = align_up(nbytes, 64);
	return reinterpret_cast<uint8_t*>(std::aligned_alloc(64, padded));
#endif
}
static inline void aligned_free64(void* p) {
#if defined(_MSC_VER)
	_aligned_free(p);
#else
	std::free(p);
#endif
}



#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>



namespace ghnsw {

using l2func_t = float(*)(const float* __restrict, const float* __restrict, std::size_t) noexcept;
inline l2func_t choose_l2() noexcept { return &ops::l2_sqr; }

using Pair = std::pair<float,uint32_t>;

struct Params {
	int M = 16;
	int ef_construction = 128;
	int ef_search = 64;
	uint64_t rng_seed = 124;
	int k_base = 2;
	std::string save_path = "index.ghnsw";
};


struct MMapRegion {
	uint8_t* data = nullptr;
	std::size_t size = 0;
	int fd = -1;
	std::size_t last_flushed_offset = 0;

	MMapRegion() = default;
	~MMapRegion() { close(); }

	void close() noexcept {
		if (data && size) {
			::msync(data, size, MS_ASYNC);
			::munmap(data, size);
		}
		data = nullptr; 
		size = 0;
		last_flushed_offset = 0;
		if (fd >= 0) ::close(fd);
		fd = -1;
	}

	void flush_all() {
		if (!data || size == 0) return;

		::madvise(data, size, MADV_DONTNEED);
		// ::posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);

		last_flushed_offset = size;
	}

	void open_readonly(const std::string& path) {
		close();
		fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0) {
			throw std::runtime_error("mmap: failed to open index file");
		}
		struct stat st{};
		if (::fstat(fd, &st) != 0) {
			::close(fd);
			throw std::runtime_error("mmap: fstat failed on index file");
		}
		if (st.st_size <= 0) {
			::close(fd);
			throw std::runtime_error("mmap: index file is empty");
		}
		size = static_cast<std::size_t>(st.st_size);
		void* p = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
		if (p == MAP_FAILED) {
			int err = errno;  // Capture errno
			::close(fd);
			fd = -1;
			throw std::runtime_error("mmap failed: " + std::string(std::strerror(err)));
		}
		data = static_cast<uint8_t*>(p);
	}

};


struct MMapRW {
	uint8_t* data = nullptr;
	std::size_t size = 0;
	int fd = -1;

	~MMapRW() { close(); }

	void close() noexcept {
		if (data && size) {
			::msync(data, size, MS_ASYNC);
			::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
			::madvise(data, size, MADV_DONTNEED);
			::munmap(data, size);
			// malloc_trim(0);
		}
		data = nullptr; size = 0;
		if (fd >= 0) ::close(fd);
		fd = -1;
	}

	void flush_async() {
		if (!data || size == 0) return;
		::msync(data, size, MS_ASYNC);
		// ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
		// ::madvise(data, size, MADV_DONTNEED);
	}

	void create_rw(const std::string& path, std::size_t bytes) {
		close();

		fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
		if (fd < 0) throw std::runtime_error("mmaprw: open failed");

		if (::ftruncate(fd, (off_t)bytes) != 0) {
			close();
			throw std::runtime_error("mmaprw: ftruncate failed");
		}
		if (bytes == 0) {
			close();
			throw std::runtime_error("mmaprw: cannot map zero bytes");
		}

		void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (p == MAP_FAILED) {
			int err = errno;
			std::cerr << "mmap failed with errno: " << err << std::endl;
			close();
			throw std::runtime_error("mmaprw failed: " + std::string(std::strerror(err)));
		}

		data = static_cast<uint8_t*>(p);
		size = bytes;
	}
};


#pragma pack(push, 1)
struct DiskHeader {
	static constexpr uint64_t kMagic   = 0x58444957534E4847ull;
	static constexpr uint32_t kVersion = 1;

	uint64_t magic;
	uint32_t version;
	uint32_t dim;
	uint64_t n;
	uint32_t M;
	uint32_t M0;
	uint32_t max_level;
	int32_t  entry_point;
	uint32_t k_base;
	uint32_t ef_search;

	uint64_t l0_offset;
	uint64_t l0_bytes;
	uint64_t l0_stride;
	uint64_t l0_off_links;
	uint64_t l0_off_vec;

	uint64_t levels_offset;
	uint64_t upper_base_offset;
	uint64_t up_deg_offset;
	uint64_t up_nbr_offset;
	uint64_t total_upper_blocks;

	uint64_t deleted_offset;
	uint64_t deleted_bytes;

	uint64_t extid_offsets_offset;
	uint64_t extid_data_offset;
	uint64_t extid_data_bytes;

	uint64_t num_inserts_since_build;
	uint64_t num_deletes_since_build;

	uint64_t reserved[4];
};
#pragma pack(pop)


struct HeapNode {
	float dist;
	uint32_t id;
};

struct BoundedMaxHeap {
	std::vector<HeapNode> a;
	uint32_t cap = 0;

	void reset(uint32_t capacity) {
		cap = capacity;
		a.clear();
		a.reserve(capacity);
	}

	inline uint32_t size() const noexcept { return (uint32_t)a.size(); }
	inline bool empty() const noexcept { return a.empty(); }

	inline float worst() const noexcept {
		return a.empty() ? std::numeric_limits<float>::infinity() : a[0].dist;
	}

	inline void sift_down(uint32_t i) noexcept {
		const uint32_t sz = (uint32_t)a.size();
		while (true) {
			uint32_t l = i * 2 + 1;
			if (l >= sz) break;
			uint32_t r = l + 1;
			uint32_t m = (r < sz && a[r].dist > a[l].dist) ? r : l;
			if (a[i].dist >= a[m].dist) break;
			std::swap(a[i], a[m]);
			i = m;
		}
	}

	inline void sift_up(uint32_t i) noexcept {
		while (i) {
			uint32_t p = (i - 1) >> 1;
			if (a[p].dist >= a[i].dist) break;
			std::swap(a[p], a[i]);
			i = p;
		}
	}

	inline bool push_if_better(float d, uint32_t id, HeapNode* evicted = nullptr) noexcept {
		if ((uint32_t)a.size() < cap) {
			a.push_back({d, id});
			sift_up((uint32_t)a.size() - 1);
			if (evicted) evicted->id = UINT32_MAX;
			return true;
		}
		if (a.empty()) { // cap might be 0
			if (evicted) evicted->id = UINT32_MAX;
			return false;
		}
		if (d >= a[0].dist) {
			if (evicted) evicted->id = UINT32_MAX;
			return false;
		}
		if (evicted) *evicted = a[0];
		a[0] = {d, id};
		sift_down(0);
		return true;
	}
};





class Index {
public:
	Index() = default;
	inline float dist(const float* __restrict a, const float* __restrict b) const noexcept {
		return l2_(a, b, d_);
	}

	Index(const std::string& vectors_path,
		  const std::vector<std::string>& external_ids,
		  std::size_t dim,
		  const Params& param,
		  const std::string& tmp_prefix = "buildtmp")
	{
		P_ = param;
		n_ = external_ids.size();
		d_ = dim;
		M0_ = (size_t)(P_.M * 2);
		mult_ = 1 / std::log(1.0 * P_.M);

		if (n_ == 0 || d_ == 0) throw std::runtime_error("fit: empty dataset");

		l2_ = choose_l2();

		external_ids_ = external_ids;
		if (external_ids_.size() != n_) {
			throw std::runtime_error("external_ids size mismatch with n");
		}

		external_to_internal_.reserve(n_);
		for (uint32_t i = 0; i < (uint32_t)n_; ++i) {
			const auto& eid = external_ids_[i];
			auto [it, inserted] = external_to_internal_.emplace(eid, i);
			if (!inserted) {
				throw std::runtime_error("duplicate external id: " + eid);
			}
		}

		deleted_.assign(n_, 0);
		num_inserts_since_build_ = 0;
		num_deletes_since_build_ = 0;

		vec_mmap_.open_readonly(vectors_path);
		const size_t need = (size_t)n_ * d_ * sizeof(float);
		if (vec_mmap_.size < need) throw std::runtime_error("vectors mmap: file too small");
		Xptr_ = reinterpret_cast<const float*>(vec_mmap_.data);
		build_disk_mode_ = true;

		level_.assign(n_, 0);
		upper_base_block_.assign(n_, 0);
		visited_tag_.assign(n_, 0);
		cur_tag_ = 1;

		level_generator_.seed(P_.rng_seed);
		precompute_levels_and_alloc_pools(tmp_prefix);
	}

	~Index() {
		if (l0_.base && l0_.owns) {
			aligned_free64(l0_.base);
			l0_.base = nullptr;
		}
	}

	// search-only constructor
	explicit Index(const std::string& index_path, int ef_search = 64) {
		l2_ = choose_l2();

		mmap_region_.open_readonly(index_path);
		if (mmap_region_.size < sizeof(DiskHeader))
		throw std::runtime_error("mmap index: file too small");

		const auto* hdr = reinterpret_cast<const DiskHeader*>(mmap_region_.data);
		if (hdr->magic != DiskHeader::kMagic)
		throw std::runtime_error("mmap index: bad magic");

		if (hdr->version != DiskHeader::kVersion)
		throw std::runtime_error("mmap index: version mismatch");

		n_          = (size_t)hdr->n;
		d_          = (size_t)hdr->dim;
		M0_         = (size_t)hdr->M0;
		max_level_  = (int)hdr->max_level;
		entry_point_= (int)hdr->entry_point;

		P_.M      = (int)hdr->M;
		P_.k_base = (int)hdr->k_base;

		visited_tag_.assign(n_, 0);
		cur_tag_ = 1;

		if (!hdr->l0_offset || !hdr->l0_bytes)
		throw std::runtime_error("mmap index: missing L0 block");
		if ((uint64_t)hdr->l0_offset + (uint64_t)hdr->l0_bytes > (uint64_t)mmap_region_.size)
		throw std::runtime_error("mmap index: L0 block out of range");

		l0_.base      = mmap_region_.data + hdr->l0_offset;
		l0_.bytes     = (size_t)hdr->l0_bytes;
		l0_.stride    = (size_t)hdr->l0_stride;
		l0_.off_links = (size_t)hdr->l0_off_links;
		l0_.off_vec   = (size_t)hdr->l0_off_vec;
		l0_.d         = hdr->dim;

		const int64_t maxm = (int64_t)hdr->M0;
		l0_.maxM = (maxm > 0) ? (uint32_t)maxm : (uint32_t)hdr->M0;



		l0_.owns  = false;
		l0_.ready = true;

		if (!hdr->levels_offset || !hdr->upper_base_offset || !hdr->up_deg_offset || !hdr->up_nbr_offset)
		throw std::runtime_error("mmap index: missing pooled upper-level offsets");

		total_upper_blocks_mmap_ = hdr->total_upper_blocks;

		auto in_range = [&](uint64_t off, uint64_t bytes) {
		return off <= (uint64_t)mmap_region_.size && bytes <= (uint64_t)mmap_region_.size - off;
		};

		const uint64_t levels_bytes = (uint64_t)n_ * sizeof(uint16_t);
		const uint64_t base_bytes   = (uint64_t)n_ * sizeof(uint32_t);
		const uint64_t deg_bytes    = (uint64_t)total_upper_blocks_mmap_ * sizeof(uint16_t);

		const uint64_t M = (uint64_t)P_.M;
		if (total_upper_blocks_mmap_ && M > (UINT64_MAX / total_upper_blocks_mmap_))
		throw std::runtime_error("mmap index: up_nbr size overflow");
		const uint64_t nbr_bytes = (uint64_t)total_upper_blocks_mmap_ * M * sizeof(uint32_t);

		if (!in_range(hdr->levels_offset, levels_bytes))       throw std::runtime_error("mmap index: levels out of range");
		if (!in_range(hdr->upper_base_offset, base_bytes))     throw std::runtime_error("mmap index: upper_base out of range");
		if (!in_range(hdr->up_deg_offset, deg_bytes))          throw std::runtime_error("mmap index: up_deg out of range");
		if (!in_range(hdr->up_nbr_offset, nbr_bytes))          throw std::runtime_error("mmap index: up_nbr out of range");

		levels_mmap_      = reinterpret_cast<const uint16_t*>(mmap_region_.data + hdr->levels_offset);
		upper_base_mmap_  = reinterpret_cast<const uint32_t*>(mmap_region_.data + hdr->upper_base_offset);
		up_deg_mmap_      = reinterpret_cast<const uint16_t*>(mmap_region_.data + hdr->up_deg_offset);
		up_nbr_mmap_      = reinterpret_cast<const uint32_t*>(mmap_region_.data + hdr->up_nbr_offset);

		P_.ef_search = (ef_search > 0) ? (uint32_t)ef_search : (uint32_t)hdr->ef_search;

		external_ids_.clear();
		external_to_internal_.clear();

		if (hdr->extid_offsets_offset && hdr->extid_data_offset) {
			const uint64_t offsets_bytes = (uint64_t)(n_ + 1) * sizeof(uint64_t);
			if (!in_range(hdr->extid_offsets_offset, offsets_bytes)) {
				throw std::runtime_error("mmap index: extid_offsets out of range");
			}
			const uint64_t* off = reinterpret_cast<const uint64_t*>(
				mmap_region_.data + hdr->extid_offsets_offset);

			const uint64_t data_bytes = hdr->extid_data_bytes;
			if (!in_range(hdr->extid_data_offset, data_bytes)) {
				throw std::runtime_error("mmap index: extid_data out of range");
			}
			const char* data = reinterpret_cast<const char*>(
				mmap_region_.data + hdr->extid_data_offset);

			if (off[n_] != data_bytes) {
				throw std::runtime_error("mmap index: extid_offsets[n] != extid_data_bytes");
			}

			external_ids_.resize(n_);
			external_to_internal_.reserve(n_);
			for (uint32_t i = 0; i < (uint32_t)n_; ++i) {
				uint64_t begin = off[i];
				uint64_t end   = off[i + 1];
				if (end < begin || end > data_bytes) {
					throw std::runtime_error("mmap index: bad extid offset range");
				}
				const size_t len = (size_t)(end - begin);
				external_ids_[i].assign(data + begin, len);

				auto [it, inserted] =
					external_to_internal_.emplace(external_ids_[i], i);
				if (!inserted) {
					throw std::runtime_error("duplicate external id in index file");
				}
			}
		} else {
			external_ids_.clear();
			external_to_internal_.clear();
		}

		num_inserts_since_build_ = hdr->num_inserts_since_build;
		num_deletes_since_build_ = hdr->num_deletes_since_build;

		deleted_.clear();

		if (hdr->deleted_offset && hdr->deleted_bytes) {
			const uint64_t bytes = hdr->deleted_bytes;
			if (bytes != (uint64_t)n_) throw std::runtime_error("mmap index: deleted_bytes != n");
			if (!in_range(hdr->deleted_offset, bytes)) throw std::runtime_error("mmap index: deleted flags out of range");

			deleted_mmap_ = mmap_region_.data + hdr->deleted_offset;
		} else {
			deleted_mmap_ = nullptr;
		}

	}

	void fit() {
		for (size_t i = 0; i < n_; ++i) {
			insert(i);
		}

		save_mmap_from_pools(P_.save_path, /*copy_vectors=*/true);
		vec_mmap_.flush_all();
		pools_.l0_deg_map.close();
		pools_.l0_nbr_map.close();
		pools_.up_deg_map.close();
		pools_.up_nbr_map.close();
		vec_mmap_.close();
	}

	BoundedMaxHeap query(const float* q_in, int topk, int efs) {
		// if (!l0_.ready) {
		// 	throw std::runtime_error("query: L0 fused layout not initialized");
		// }
		thread_local QueryWorkspace ws;
		topk = (topk > 0 ? (uint32_t)topk : 0);
		const uint32_t ef = std::max<uint32_t>(efs, topk);
		ws.ensure_size(n_);
		ws.next_tag();
		ws.prepare_heaps(ef);
		uint32_t ep = entry_point_;
		float cur_dist = l2_(vec0(ep), q_in, d_);

		for (int l = max_level_; l >= 0; --l) {
			bool changed = true;
			while (changed) {
				changed = false;
				auto [beg, end] = fetch_neighbors(l, ep);
				const uint32_t* p = beg;
				for (; p != end; ++p) {
					if (p + 4 < end) {
						GHNSW_PREFETCH(vec0(*(p + 4)));
					}
					const uint32_t nb = *p;
					float d_nb = l2_(q_in, vec0(nb), d_);
					if (d_nb < cur_dist) {
						cur_dist = d_nb;
						ep = nb;
						changed = true;
					}
				}
			}
		}
		uint32_t alive_in_best = 0;

		ws.cand_heap.push_back({cur_dist, ep});
		std::push_heap(ws.cand_heap.begin(), ws.cand_heap.end(), MinByFirst{});

		ws.best.push_if_better(cur_dist, ep);

		if (!is_deleted(ep)) {
			++alive_in_best;
		}

		ws.mark_visited(ep);
		float worst_best = cur_dist;

		while (!ws.cand_heap.empty()) {
			if (alive_in_best >= topk && ws.best.size() >= ef && ws.cand_heap.front().first > worst_best) break;

			std::pop_heap(ws.cand_heap.begin(), ws.cand_heap.end(), MinByFirst{});
			const uint32_t cand = ws.cand_heap.back().second;
			ws.cand_heap.pop_back();

			auto [beg, end] = neighbors0(cand);

			if (beg != end) {
				GHNSW_PREFETCH(beg);
				if (beg + 16 < end) GHNSW_PREFETCH(beg + 16);
				const uint32_t nb0 = *beg;
				GHNSW_PREFETCH(ws.visited.data() + nb0);
				GHNSW_PREFETCH(vec0(nb0));
			}

			for (const uint32_t* p = beg; p != end; ++p) {
				if (p + 1 < end) {
					const uint32_t nb_next = *(p + 1);
					GHNSW_PREFETCH(ws.visited.data() + nb_next);
					GHNSW_PREFETCH(vec0(nb_next));
				}

				const uint32_t nb = *p;
				if (ws.is_visited(nb)) continue;
				ws.mark_visited(nb);

				const float d_nb = l2_(q_in, vec0(nb), l0_.d);

				if (ws.best.size() < ef || d_nb < worst_best) {
					ws.cand_heap.push_back({d_nb, nb});
					std::push_heap(ws.cand_heap.begin(), ws.cand_heap.end(), MinByFirst{});

					const bool nb_dead = is_deleted(nb);

					if (!nb_dead || (alive_in_best >= topk)) {
						HeapNode ev;
						const bool accepted = ws.best.push_if_better(d_nb, nb, &ev);

						if (accepted) {
							if (!nb_dead) ++alive_in_best;

							// if an eviction happened (heap was full and we replaced the root)
							if (ev.id != UINT32_MAX) {
								if (!is_deleted(ev.id) && alive_in_best > 0) {
									--alive_in_best;
								}
							}
							worst_best = ws.best.worst();
						}
					}

				}
			}
		}
		return ws.best;
		// std::vector<Pair> res;
		// res.reserve(ws.best.size());
		// res.insert(res.end(), ws.best_heap.begin(), ws.best_heap.end());
		// std::sort(res.begin(), res.end(), [](const Pair& a, const Pair& b){ return a.first < b.first; });
		// out_idx.clear();
		// out_idx.reserve(topk);

		// for (const auto& [dist, id] : res) {
		// 	if (!is_deleted(id)) {
		// 		out_idx.push_back(external_id(id));
		// 		if ((int)out_idx.size() >= topk) break;
		// 	}
		// }
	}

	struct QueryWorkspace {
		std::vector<uint32_t> visited;   // [n]
		uint32_t tag = 1;

		std::vector<Pair> cand_heap;     // min-heap by distance
		BoundedMaxHeap best;             // max-heap by distance

		void ensure_size(std::size_t n) {
			if (visited.size() != n) {
				visited.assign(n, 0);
				tag = 1;
			}
		}

		inline void next_tag() {
			++tag;
			if (tag == 0) {
				std::fill(visited.begin(), visited.end(), 0);
				tag = 1;
			}
		}

		inline bool is_visited(uint32_t u) const noexcept { return visited[u] == tag; }
		inline void mark_visited(uint32_t u) noexcept { visited[u] = tag; }

		inline void prepare_heaps(uint32_t ef) {
			cand_heap.clear();
			if (cand_heap.capacity() < ef) cand_heap.reserve(ef);
			best.reset(ef);
		}
	};


	const Params& params() const { return P_; }
	int dim() const { return static_cast<int>(d_); }
	std::size_t size() const noexcept { return n_; }
	std::size_t active_size() const noexcept {
		size_t alive = 0;
		for (uint32_t i = 0; i < (uint32_t)n_; ++i) {
			if (!is_deleted(i)) ++alive;
		}
		return alive;
	}
	int max_level() const noexcept { return max_level_; }


	std::size_t mark_deleted_by_external_ids_and_persist(
		const std::string& index_path,
		const std::vector<std::string>& external_ids,
		std::vector<std::string>* not_found = nullptr)
	{
		std::vector<uint32_t> internal_ids;
		internal_ids.reserve(external_ids.size());

		for (const auto& eid : external_ids) {
			auto it = external_to_internal_.find(eid);
			if (it == external_to_internal_.end()) {
				if (not_found) not_found->push_back(eid);
				continue;
			}
			internal_ids.push_back(it->second);
		}

		if (internal_ids.empty()) return 0;

		std::sort(internal_ids.begin(), internal_ids.end());
		internal_ids.erase(std::unique(internal_ids.begin(), internal_ids.end()), internal_ids.end());


		std::size_t newly_deleted = persist_deleted_and_counters_inplace(index_path, internal_ids);

		if (!deleted_mmap_) {
			for (uint32_t id : internal_ids) {
				if (id < deleted_.size()) deleted_[id] = 1;
			}
		}

		num_deletes_since_build_ += newly_deleted;
		return newly_deleted;
	}


	std::size_t persist_deleted_and_counters_inplace(
		const std::string& index_path,
		const std::vector<uint32_t>& internal_ids) const
	{
		int fd = ::open(index_path.c_str(), O_RDWR | O_CLOEXEC);
		if (fd < 0) throw std::runtime_error("persist: open failed");

		if (::flock(fd, LOCK_EX) != 0) { ::close(fd); throw std::runtime_error("persist: flock failed"); }

		DiskHeader hdr{};
		if (::pread(fd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
			::flock(fd, LOCK_UN); ::close(fd);
			throw std::runtime_error("persist: pread header failed");
		}

		if (hdr.magic != DiskHeader::kMagic || hdr.version != DiskHeader::kVersion) {
			::flock(fd, LOCK_UN); ::close(fd);
			throw std::runtime_error("persist: bad header");
		}

		if (hdr.deleted_offset == 0 || hdr.deleted_bytes != (uint64_t)n_) {
			::flock(fd, LOCK_UN); ::close(fd);
			throw std::runtime_error("persist: deleted region mismatch");
		}

		std::size_t newly_deleted = 0;

		for (uint32_t id : internal_ids) {
			if (id >= n_) continue;
			const uint64_t off = hdr.deleted_offset + (uint64_t)id;

			uint8_t cur = 0;
			if (::pread(fd, &cur, 1, (off_t)off) != 1) {
				::flock(fd, LOCK_UN); ::close(fd);
				throw std::runtime_error("persist: pread deleted byte failed");
			}

			if (cur == 0) {
				uint8_t one = 1;
				if (::pwrite(fd, &one, 1, (off_t)off) != 1) {
					::flock(fd, LOCK_UN); ::close(fd);
					throw std::runtime_error("persist: pwrite deleted byte failed");
				}
				++newly_deleted;
			}
		}

		if (::fdatasync(fd) != 0) {
			::flock(fd, LOCK_UN); ::close(fd);
			throw std::runtime_error("persist: fdatasync failed");
		}

		hdr.num_deletes_since_build += newly_deleted;

		if (::pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
			::flock(fd, LOCK_UN); ::close(fd);
			throw std::runtime_error("persist: pwrite header failed");
		}

		if (::fsync(fd) != 0) {
			::flock(fd, LOCK_UN); ::close(fd);
			throw std::runtime_error("persist: fsync failed");
		}

		::flock(fd, LOCK_UN);
		::close(fd);
		return newly_deleted;
	}


	uint64_t num_inserts_since_build() const noexcept {
		return num_inserts_since_build_;
	}

	uint64_t num_deletes_since_build() const noexcept {
		return num_deletes_since_build_;
	}

	void set_num_inserts_since_build(int n) noexcept {
		// this should be permanent
		num_inserts_since_build_ += n;
	}

	void set_num_deletes_since_build(int n) noexcept {
		// this should be permanent
		num_deletes_since_build_ += n;
	}


	inline bool is_deleted(uint32_t id) const noexcept {
		if (id >= n_) return true;
		if (deleted_mmap_) return deleted_mmap_[id] != 0;
		if (deleted_.size() == n_) return deleted_[id] != 0;
		return false;
	}

	static inline void fsync_dir_of(const std::string& path) {
		auto slash = path.find_last_of('/');
		std::string dir = (slash == std::string::npos) ? "." : path.substr(0, slash);
		int dfd = ::open(dir.c_str(), O_DIRECTORY | O_RDONLY);
		if (dfd >= 0) { ::fsync(dfd); ::close(dfd); }
	}

	static inline void atomic_replace_file(const std::string& tmp, const std::string& dst) {
		if (::rename(tmp.c_str(), dst.c_str()) != 0)
			throw std::runtime_error("rename failed: " + std::string(std::strerror(errno)));
		fsync_dir_of(dst);
	}

	const uint8_t* deleted_mmap_ = nullptr;

	const std::vector<std::string>& external_ids() const noexcept { return external_ids_; }
	const std::string& external_id(uint32_t internal) const {
		if (internal >= n_) throw std::runtime_error("external_id: OOB");
		return external_ids_.at(internal);
	}

	void export_alive_vectors_to_file(
		const std::string& out_vectors_path,
		std::vector<std::string>& out_external_ids) const
	{
		if (!l0_.ready) throw std::runtime_error("export_alive_vectors_to_file: L0 not ready");
		if (d_ == 0) throw std::runtime_error("export_alive_vectors_to_file: dim=0");

		size_t alive = 0;
		for (uint32_t i = 0; i < (uint32_t)n_; ++i) if (!is_deleted(i)) ++alive;

		int fd = ::open(out_vectors_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
		if (fd < 0) throw std::runtime_error("export: open failed");

		const size_t out_bytes = alive * d_ * sizeof(float);
		if (::ftruncate(fd, (off_t)out_bytes) != 0) { ::close(fd); throw std::runtime_error("export: ftruncate failed"); }

		if (out_bytes == 0) {
			::close(fd);
			throw std::runtime_error("export: cannot map zero bytes");
		}
		void* p = ::mmap(nullptr, out_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (p == MAP_FAILED) {
			::close(fd);
			int err = errno;
			std::cerr << "mmap failed with errno: " << err << std::endl;
			throw std::runtime_error("export: mmap failed: " + std::string(std::strerror(err)));
		}

		float* dst = reinterpret_cast<float*>(p);

		out_external_ids.clear();
		out_external_ids.reserve(alive);

		size_t w = 0;
		for (uint32_t i = 0; i < (uint32_t)n_; ++i) {
			if (is_deleted(i)) continue;
			std::memcpy(dst + w * d_, vec0(i), d_ * sizeof(float));
			out_external_ids.push_back(external_ids_.at(i));
			++w;
		}

		::msync(p, out_bytes, MS_SYNC);
		::munmap(p, out_bytes);
		::close(fd);
		fsync_dir_of(out_vectors_path);
	}

private:


	static inline uint8_t pread1(int fd, uint64_t off) {
	  uint8_t v;
	  if (::pread(fd, &v, 1, (off_t)off) != 1) throw std::runtime_error("pread1 failed");
	  return v;
	}


	std::vector<std::string> external_ids_;

	std::unordered_map<std::string, uint32_t> external_to_internal_;

	std::vector<uint8_t> deleted_;

	uint64_t num_inserts_since_build_ = 0;
	uint64_t num_deletes_since_build_ = 0;

	MMapRegion vec_mmap_;
	const float* Xptr_ = nullptr;
	bool build_disk_mode_ = false;

	struct BuildPools {
		MMapRW l0_deg_map, l0_nbr_map;
		MMapRW up_deg_map, up_nbr_map;

		uint16_t* l0_deg = nullptr;   // [n]
		uint32_t* l0_nbr = nullptr;   // [n * M0]

		uint16_t* up_deg = nullptr;   // [total_upper_blocks]
		uint32_t* up_nbr = nullptr;   // [total_upper_blocks * M]

		uint64_t total_upper_blocks = 0;
	} pools_;

	inline const float* vecX(uint32_t id) const noexcept {
		return Xptr_ + (size_t)id * d_;
	}

	std::vector<uint32_t> upper_base_block_;

	void precompute_levels_and_alloc_pools(const std::string& tmp_prefix) {

		uint64_t total_blocks = 0;
		int maxL = 0;

		for (size_t i = 0; i < n_; ++i) {
			int lv;
			if (preset_levels_.size() == n_) {
				lv = std::max(0, preset_levels_[i]);
			} else {
				lv = get_random_level(mult_);
			}
			level_[i] = lv;
			maxL = std::max(maxL, lv);

			upper_base_block_[i] = (uint32_t)total_blocks;
			total_blocks += (lv >= 1) ? (uint64_t)lv : 0;
		}

		pools_.total_upper_blocks = total_blocks;

		pools_.l0_deg_map.create_rw(tmp_prefix + ".l0deg", n_ * sizeof(uint16_t));
		pools_.l0_nbr_map.create_rw(tmp_prefix + ".l0nbr", (size_t)n_ * M0_ * sizeof(uint32_t));

		pools_.l0_deg = reinterpret_cast<uint16_t*>(pools_.l0_deg_map.data);
		pools_.l0_nbr = reinterpret_cast<uint32_t*>(pools_.l0_nbr_map.data);

		std::memset(pools_.l0_deg, 0, n_ * sizeof(uint16_t));

		if (total_blocks > 0) {
			pools_.up_deg_map.create_rw(tmp_prefix + ".updeg", (size_t)total_blocks * sizeof(uint16_t));
			pools_.up_nbr_map.create_rw(tmp_prefix + ".upnbr", (size_t)total_blocks * (size_t)P_.M * sizeof(uint32_t));
			pools_.up_deg = reinterpret_cast<uint16_t*>(pools_.up_deg_map.data);
			pools_.up_nbr = reinterpret_cast<uint32_t*>(pools_.up_nbr_map.data);
			std::memset(pools_.up_deg, 0, (size_t)total_blocks * sizeof(uint16_t));
		} else {
			pools_.up_deg = nullptr;
			pools_.up_nbr = nullptr;
		}

		max_level_ = -1;
		entry_point_ = -1;
	}

	void insert(std::size_t i){
		int clevel = level_[i];

		if (entry_point_ == -1) {
			entry_point_ = static_cast<int>(i);
			max_level_ = clevel;
			return;
		}
		const float* qi = vecX((uint32_t)i);

		uint32_t ep = static_cast<uint32_t>(entry_point_);
		float cur_dist = l2_(qi, vecX(ep), d_);
		for (int l = max_level_; l > clevel; --l)
		{
			bool changed = true;
			while (changed) {
				changed = false;
				auto [beg, end] = neighbors_build(ep, l);
				for (const uint32_t* p = beg; p != end; ++p) {
					uint32_t nb = *p;
					if (is_deleted(nb)) continue;
					float d = l2_(qi, vecX(nb), d_);
					if (d < cur_dist) {
						cur_dist = d; ep = nb; changed = true;
					}
				}
			}
		}
		for (int l = std::min(clevel, max_level_); l >= 0; --l)
		{
			++cur_tag_;
			if (cur_tag_ == 0) {
				std::fill(visited_tag_.begin(), visited_tag_.end(), 0);
				cur_tag_ = 1;
			}
			auto mark_visited = [&](uint32_t u) {
				visited_tag_[u] = cur_tag_;
			};
			auto is_visited = [&](uint32_t u) -> bool {
				return visited_tag_[u] == cur_tag_;
			};

			std::priority_queue<Pair> cand_heap;
			std::priority_queue<Pair> best_heap;
			mark_visited(ep);
			cur_dist = l2_(qi, vecX(ep), d_);
			cand_heap.emplace(-cur_dist, ep);
			best_heap.emplace(cur_dist, ep);
			float worst_best = cur_dist;
			while (!cand_heap.empty()) {
				float bestCandDist = -cand_heap.top().first; // convert back to +dist
				if (best_heap.size() == (size_t)P_.ef_construction && bestCandDist > worst_best)
					break;
				// if (cand_heap.top().first > worst_best && best_heap.size() == P_.ef_construction) {
				// 	break;
				// }
				auto [d_cand, cand] = cand_heap.top(); cand_heap.pop(); d_cand = -d_cand;

				auto [beg, end] = neighbors_build(cand, l);
				for (const uint32_t* p = beg; p != end; ++p) {
					uint32_t nb = *p;
					if (is_visited(nb) or is_deleted(nb)) continue;
					mark_visited(nb);
					float d_nb = l2_(qi, vecX(nb), d_);
					if (best_heap.size() < P_.ef_construction || worst_best > d_nb) {
						cand_heap.emplace(-d_nb, nb);
						best_heap.emplace(d_nb, nb);
						if (best_heap.size() > P_.ef_construction) {
							best_heap.pop();
						}
						worst_best = best_heap.top().first;
					}
				}
			}
			std::vector<Pair> candlist;
			candlist.reserve(best_heap.size());
			while (!best_heap.empty()) {
				candlist.push_back(best_heap.top());
				best_heap.pop();
			}
			std::sort(candlist.begin(), candlist.end(), [](const Pair& a, const Pair& b){
				return a.first < b.first;
			});
			ep = candlist.front().second;
			const std::size_t cap = (l == 0) ? M0_ : static_cast<std::size_t>(P_.M);
			std::vector<unsigned int> cids;
			cids.reserve(candlist.size());
			for (auto& p : candlist) cids.push_back(p.second);

			auto neighbors = heuristic2(static_cast<uint32_t>(i), cids, cap);
			link_bidirectional(static_cast<unsigned int>(i), neighbors, l);
		}
		if (clevel > max_level_) {
			entry_point_ = static_cast<int>(i);
			max_level_ = clevel;
		}
	}

	void save_mmap_from_pools(const std::string& path, bool copy_vectors) const {
		if (entry_point_ < 0) throw std::runtime_error("save: empty index");

		DiskHeader hdr{};
		hdr.magic       = DiskHeader::kMagic;
		hdr.version     = DiskHeader::kVersion;
		hdr.dim         = (uint32_t)d_;
		hdr.n           = (uint64_t)n_;
		hdr.M           = (uint32_t)P_.M;
		hdr.M0          = (uint32_t)M0_;
		hdr.max_level   = (uint32_t)max_level_;
		hdr.entry_point = (int32_t)entry_point_;
		hdr.k_base      = (uint32_t)P_.k_base;
		hdr.ef_search   = (uint32_t)P_.ef_search;
		hdr.total_upper_blocks = pools_.total_upper_blocks;

		hdr.num_inserts_since_build = num_inserts_since_build_;
		hdr.num_deletes_since_build = num_deletes_since_build_;

		std::vector<uint64_t> extid_offsets;
		uint64_t extid_data_bytes = 0;
		if (!external_ids_.empty()) {
			if (external_ids_.size() != n_) {
				throw std::runtime_error("save: external_ids_ size mismatch with n");
			}
			extid_offsets.resize(n_ + 1);
			uint64_t cur = 0;
			for (size_t i = 0; i < n_; ++i) {
				extid_offsets[i] = cur;
				cur += (uint64_t)external_ids_[i].size();
			}
			extid_offsets[n_] = cur;
			extid_data_bytes = cur;
		}
		hdr.extid_data_bytes = extid_data_bytes;

		const uint32_t maxM0 = (uint32_t)(M0_);
		size_t links_bytes = sizeof(uint32_t) + (size_t)maxM0 * sizeof(uint32_t);
		size_t links_aln   = align_up(links_bytes, 64);
		size_t vec_bytes   = copy_vectors ? align_up((size_t)d_ * sizeof(float), 64) : 0;

		hdr.l0_off_links = 0;
		hdr.l0_off_vec   = links_aln;
		hdr.l0_stride    = links_aln + vec_bytes;
		hdr.l0_bytes     = (uint64_t)((size_t)n_ * (size_t)hdr.l0_stride);

		size_t offset = align_up(sizeof(DiskHeader), 4096);
		hdr.l0_offset = offset;
		offset = align_up(offset + (size_t)hdr.l0_bytes, 4096);

		// levels[n]
		hdr.levels_offset = offset;
		offset += n_ * sizeof(uint16_t);            // or uint8_t
		offset = align_up(offset, 4096);

		// upper_base_block[n]
		hdr.upper_base_offset = offset;
		offset += n_ * sizeof(uint32_t);
		offset = align_up(offset, 4096);

		// up_deg[total_blocks]
		hdr.up_deg_offset = offset;
		offset += (size_t)hdr.total_upper_blocks * sizeof(uint16_t);
		offset = align_up(offset, 4096);

		// up_nbr[total_blocks * M]
		hdr.up_nbr_offset = offset;
		offset += (size_t)hdr.total_upper_blocks * (size_t)P_.M * sizeof(uint32_t);
		offset = align_up(offset, 4096);

		// deleted[n] byte flags
		hdr.deleted_offset = offset;
		hdr.deleted_bytes  = (uint64_t)n_;
		offset += n_;
		offset = align_up(offset, 4096);

		// extid_offsets[n+1] (if any)
		if (!external_ids_.empty()) {
			hdr.extid_offsets_offset = offset;
			offset += (size_t)(n_ + 1) * sizeof(uint64_t);
			offset = align_up(offset, 4096);

			// extid_data
			hdr.extid_data_offset = offset;
			offset += (size_t)hdr.extid_data_bytes;
			offset = align_up(offset, 4096);
		} else {
			hdr.extid_offsets_offset = 0;
			hdr.extid_data_offset   = 0;
		}

		const size_t total_size = offset;

		MMapRW out;
		out.create_rw(path, total_size);
		uint8_t* base = out.data;

		std::memcpy(base, &hdr, sizeof(DiskHeader));

		for (uint32_t id = 0; id < (uint32_t)n_; ++id) {
			uint8_t* node = base + (size_t)hdr.l0_offset + (size_t)id * (size_t)hdr.l0_stride;

			uint32_t deg = (uint32_t)std::min<size_t>(pools_.l0_deg[id], maxM0);
			*reinterpret_cast<uint32_t*>(node + hdr.l0_off_links) = deg;

			uint32_t* ids = reinterpret_cast<uint32_t*>(node + hdr.l0_off_links + sizeof(uint32_t));
			if (deg) {
				const uint32_t* src = pools_.l0_nbr + (size_t)id * M0_;
				std::memcpy(ids, src, (size_t)deg * sizeof(uint32_t));
			}

			if (copy_vectors) {
				float* dstv = reinterpret_cast<float*>(node + hdr.l0_off_vec);
				const float* srcv = vecX(id);
				std::memcpy(dstv, srcv, (size_t)d_ * sizeof(float));
			}
		}

		auto* lvl = reinterpret_cast<uint16_t*>(base + hdr.levels_offset);
		for (size_t i = 0; i < n_; ++i) {
			int v = level_[i];
			if (v < 0) v = 0;
			if (v > 65535) throw std::runtime_error("level too large");
			lvl[i] = (uint16_t)v;
		}

		// upper_base_block
		std::memcpy(base + hdr.upper_base_offset,
				  upper_base_block_.data(),
				  n_ * sizeof(uint32_t));

		std::memcpy(base + hdr.up_deg_offset,
				  pools_.up_deg,
				  (size_t)hdr.total_upper_blocks * sizeof(uint16_t));

		std::memcpy(base + hdr.up_nbr_offset,
				  pools_.up_nbr,
				  (size_t)hdr.total_upper_blocks * (size_t)P_.M * sizeof(uint32_t));

		if (!deleted_.empty()) {
			if (deleted_.size() != n_) {
				throw std::runtime_error("save: deleted_ size mismatch with n");
			}
			std::memcpy(base + hdr.deleted_offset,
						deleted_.data(),
						n_);
		}

		if (!external_ids_.empty()) {
			std::memcpy(base + hdr.extid_offsets_offset,
						extid_offsets.data(),
						(size_t)(n_ + 1) * sizeof(uint64_t));

			char* id_data = reinterpret_cast<char*>(base + hdr.extid_data_offset);
			for (size_t i = 0; i < n_; ++i) {
				const auto& s = external_ids_[i];
				if (!s.empty()) {
					std::memcpy(id_data + extid_offsets[i],
								s.data(),
								s.size());
				}
			}
		}
		::msync(base, total_size, MS_SYNC);
	}

	inline int node_level(uint32_t u) const noexcept {
		if (levels_mmap_) return (int)levels_mmap_[u];
		return level_[u];
	}

	inline std::pair<const uint32_t*, const uint32_t*>
	fetch_neighbors(int level, uint32_t u) const {
		if (level == 0 && l0_.ready) return neighbors0(u);

		const int lv = node_level(u);
		if (level > lv) return {nullptr, nullptr};  // empty

		const uint32_t base = levels_mmap_ ? upper_base_mmap_[u] : upper_base_block_[u];
		const uint64_t block = (uint64_t)base + (uint64_t)(level - 1);


		const uint16_t deg = levels_mmap_ ? up_deg_mmap_[block] : pools_.up_deg[block];
		const uint32_t* nbr = (levels_mmap_)
		  ? (up_nbr_mmap_ + block * (uint64_t)P_.M)
		  : (pools_.up_nbr + block * (uint64_t)P_.M);

		const uint16_t d = (deg > (uint16_t)P_.M) ? (uint16_t)P_.M : deg;
		return { nbr, nbr + d };
	}


	inline const float* vec0(uint32_t id) const {
		return reinterpret_cast<const float*>(l0_.base + (size_t)id * l0_.stride + l0_.off_vec);
	}

	inline std::pair<const uint32_t*, const uint32_t*> neighbors0(uint32_t id) const {
		const uint32_t* beg = reinterpret_cast<const uint32_t*>(l0_.base + (size_t)id * l0_.stride + l0_.off_links + sizeof(uint32_t));
		return { beg, beg + *reinterpret_cast<const uint32_t*>(l0_.base + (size_t)id * l0_.stride + l0_.off_links) };
	}

	int get_random_level(double reverse_size) {
		std::uniform_real_distribution<double> distribution(0.0, 1.0);
		double r = -log(distribution(level_generator_)) * reverse_size;
		return (int)r;
	}

	inline void set_neighbors_block(uint32_t u, int l, const std::vector<uint32_t>& ns) {
		const size_t cap = cap_for_level(l);
		const uint16_t m = (uint16_t)std::min(ns.size(), cap);

		uint32_t* p = nbr_ptr(u, l);
		if (m) std::memcpy(p, ns.data(), (size_t)m * sizeof(uint32_t));
		deg_ref(u, l) = m;
	}

	inline void push_unique_prune(uint32_t u, int l, uint32_t x) {
		if (x == u) return;
		uint16_t& deg = deg_ref(u, l);
		uint32_t* p = nbr_ptr(u, l);
		const size_t cap = cap_for_level(l);

		for (uint16_t t = 0; t < deg; ++t) if (p[t] == x) return;

		if (deg < cap) {
			p[deg++] = x;
			return;
		}

		std::vector<uint32_t> tmp;
		tmp.reserve((size_t)deg + 1);
		tmp.insert(tmp.end(), p, p + deg);
		tmp.push_back(x);
		auto pruned = heuristic2(u, tmp, cap);
		set_neighbors_block(u, l, pruned);
	}

	void link_bidirectional(uint32_t src, const std::vector<uint32_t>& dests, int level) {
		for (uint32_t d : dests) {
			if (d == src) continue;
			if (is_deleted(d)) continue;
			push_unique_prune(src, level, d);
			push_unique_prune(d, level, src);
		}
	}

	std::vector<uint32_t> heuristic2(uint32_t query_id,
									 const std::vector<uint32_t>& candidates,
									 std::size_t m) const
	{
		std::vector<uint32_t> selected;
		if (m == 0 || candidates.empty()) return selected;

		std::vector<Pair> scored;
		scored.reserve(candidates.size());

		const float* qv = vecX(query_id);

		for (uint32_t cid : candidates) {
			if (cid == query_id) continue;
			if (is_deleted(cid)) continue;
			const float dq = l2_(vecX(cid), qv, d_);
			scored.push_back({dq, cid});
		}

		std::sort(scored.begin(), scored.end(),
				  [](const Pair& a, const Pair& b) { return a.first < b.first; });

		selected.reserve(std::min(m, scored.size()));

		for (const auto& [dq_c, cid] : scored) {
			const float* cv = vecX(cid);

			bool good = true;
			for (uint32_t sid : selected) {
				const float dc_s = l2_(cv, vecX(sid), d_);
				if (dc_s < dq_c) { good = false; break; }
			}

			if (good) {
				selected.push_back(cid);
				if (selected.size() == m) break;
			}
		}
		return selected;
	}


	Params P_;
	std::size_t n_ = 0;
	std::size_t d_ = 0;
	std::size_t M0_ = 0;
	double mult_{0.0};

	l2func_t l2_ = nullptr;


	mutable std::vector<uint32_t> visited_tag_;
	mutable uint32_t cur_tag_ = 0;


	std::vector<int> level_;
	int max_level_ = -1;
	int entry_point_ = -1;


	std::mt19937_64 level_generator_;

	std::vector<int> preset_levels_;

	struct MinByFirst { bool operator()(const Pair& a, const Pair& b) const { return a.first > b.first; } }; // min-heap
	struct MaxByFirst { bool operator()(const Pair& a, const Pair& b) const { return a.first < b.first; } }; // max-heap

	struct L0Fused {
		uint8_t* base = nullptr;
		size_t   bytes = 0;
		size_t   stride = 0;
		size_t   off_links = 0;
		size_t   off_vec   = 0;
		uint32_t maxM = 0;
		uint32_t d = 0;
		bool     ready = false;
		bool     owns  = false;
	} l0_;

	MMapRegion mmap_region_;

	inline size_t cap_for_level(int l) const noexcept {
		return (l == 0) ? M0_ : (size_t)P_.M;
	}

	inline uint16_t& deg_ref(uint32_t u, int l) noexcept {
		if (l == 0) return pools_.l0_deg[u];
		const uint32_t base = upper_base_block_[u];
		return pools_.up_deg[ base + (uint32_t)(l - 1) ];
	}

	inline const uint16_t& deg_ref(uint32_t u, int l) const noexcept {
		if (l == 0) return pools_.l0_deg[u];
		const uint32_t base = upper_base_block_[u];
		return pools_.up_deg[ base + (uint32_t)(l - 1) ];
	}

	inline uint32_t* nbr_ptr(uint32_t u, int l) noexcept {
		if (l == 0) return pools_.l0_nbr + (size_t)u * M0_;
		const uint32_t base = upper_base_block_[u];
		return pools_.up_nbr + (size_t)(base + (uint32_t)(l - 1)) * (size_t)P_.M;
	}

	inline const uint32_t* nbr_ptr(uint32_t u, int l) const noexcept {
		if (l == 0) return pools_.l0_nbr + (size_t)u * M0_;
		const uint32_t base = upper_base_block_[u];
		return pools_.up_nbr + (size_t)(base + (uint32_t)(l - 1)) * (size_t)P_.M;
	}


	inline std::pair<const uint32_t*, const uint32_t*> neighbors_build(uint32_t u, int l) const noexcept {
		uint16_t deg = (l == 0) ? pools_.l0_deg[u]
								: pools_.up_deg[ upper_base_block_[u] + (uint32_t)(l - 1) ];
		const uint32_t* p = (l == 0) ? (pools_.l0_nbr + (size_t)u * M0_)
									 : (pools_.up_nbr + (size_t)(upper_base_block_[u] + (uint32_t)(l - 1)) * (size_t)P_.M);
		return {p, p + deg};
	}

	const uint16_t* levels_mmap_ = nullptr;       // [n]
	const uint32_t* upper_base_mmap_ = nullptr;   // [n]
	const uint16_t* up_deg_mmap_ = nullptr;       // [total_blocks]
	const uint32_t* up_nbr_mmap_ = nullptr;       // [total_blocks * M]
	uint64_t total_upper_blocks_mmap_ = 0;

};

}
