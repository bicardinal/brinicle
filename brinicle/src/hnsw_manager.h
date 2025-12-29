#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <errno.h>
#include "hnsw.h"

namespace ghnsw_mgr {

// small fs helpers
static inline bool file_exists(const std::string& p) {
	struct stat st{};
	return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static inline void unlink_noexcept(const std::string& p) noexcept {
	::unlink(p.c_str());
}

static inline time_t file_mtime(const std::string& p) {
	struct stat st{};
	if (::stat(p.c_str(), &st) == 0) {
		return st.st_mtime;
	}
	return 0;
}

static inline std::string unique_temp_file(const std::string& base_path, 
										  const std::string& suffix = ".tmp") {
	const auto now = std::chrono::high_resolution_clock::now()
							.time_since_epoch().count();
	return base_path + suffix + "." + 
		   std::to_string((long long)getpid()) + "." + 
		   std::to_string((long long)now);
}

static inline void cleanup_build_pools(const std::string& tmp_prefix) {
	unlink_noexcept(tmp_prefix + ".l0deg");
	unlink_noexcept(tmp_prefix + ".l0nbr");
	unlink_noexcept(tmp_prefix + ".updeg");
	unlink_noexcept(tmp_prefix + ".upnbr");
}

static inline void atomic_rename(const std::string& oldpath, const std::string& newpath) {
	if (::rename(oldpath.c_str(), newpath.c_str()) != 0) {
		int err = errno;
		throw std::runtime_error(
			"atomic_rename failed: " + oldpath + " -> " + newpath + 
			" (errno " + std::to_string(err) + ": " + std::strerror(err) + ")"
		);
	}
	// durability
	ghnsw::Index::fsync_dir_of(newpath);
}

class FileLock {
public:
	FileLock(const std::string& lock_path) 
	: path_(lock_path), fd_(-1) {
		fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
		if (fd_ < 0) {
			throw std::runtime_error("FileLock: failed to open lock file: " + path_);
		}

		int attempts = 0;
		const int max_attempts = 6000; // 600 seconds

		while (attempts < max_attempts) {
			if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
				locked_ = true;
				return;
			}

			if (errno != EWOULDBLOCK && errno != EAGAIN) {
				::close(fd_);
				fd_ = -1;
				throw std::runtime_error("FileLock: flock failed: " + path_);
			}

			usleep(100000);
			++attempts;
		}

		// timeout
		::close(fd_);
		fd_ = -1;
		throw std::runtime_error("FileLock: timeout waiting for lock (another process writing): " + path_);
	}

	~FileLock() {
		unlock();
	}

	void unlock() noexcept {
		if (fd_ >= 0 && locked_) {
			::flock(fd_, LOCK_UN);
			::close(fd_);
			fd_ = -1;
			locked_ = false;
		}
	}

	FileLock(const FileLock&) = delete;
	FileLock& operator=(const FileLock&) = delete;

private:
	std::string path_;
	int fd_;
	bool locked_ = false;
};

class CombinedLock {
public:
	explicit CombinedLock(const std::string& lock_path)
		: mtx_(get_mutex(lock_path)), guard_(*mtx_), flock_(lock_path) {}

private:
	static std::shared_ptr<std::mutex> get_mutex(const std::string& p) {
		static std::mutex map_mu;
		static std::unordered_map<std::string, std::weak_ptr<std::mutex>> m;
		std::lock_guard<std::mutex> lk(map_mu);
		auto it = m.find(p);
		if (it != m.end()) {
			if (auto sp = it->second.lock()) return sp;
		}
		auto sp = std::make_shared<std::mutex>();
		m[p] = sp;
		return sp;
	}

	std::shared_ptr<std::mutex> mtx_;
	std::unique_lock<std::mutex> guard_;
	FileLock flock_;
};



class VectorIngestWriter {
public:
	void init(const std::string& vectors_path, std::size_t dim, bool truncate = true) {
		close();

		path_ = vectors_path;
		dim_ = dim;
		n_ = 0;
		ids_.clear();

		int flags = O_RDWR | O_CREAT;
		if (truncate) flags |= O_TRUNC;

		fd_ = ::open(path_.c_str(), flags, 0644);
		if (fd_ < 0) {
			int error_code = errno;
			std::string error_message = "VectorIngestWriter: Failed to open file '" + path_ + "'. ";
			error_message += "Error code: " + std::to_string(error_code) + " (" + std::strerror(error_code) + "). ";
			switch (error_code) {
				case EACCES:
					error_message += "Permission denied. Check file/directory permissions.";
					break;
				case ENOENT:
					error_message += "File or directory does not exist. Verify the path is correct.";
					break;
				case ENOTDIR:
					error_message += "A component of the path is not a directory.";
					break;
				case EISDIR:
					error_message += "The path refers to a directory, not a file.";
					break;
				case EROFS:
					error_message += "The filesystem is read-only.";
					break;
				case EMFILE:
					error_message += "Too many files open in this process.";
					break;
				case ENFILE:
					error_message += "Too many files open system-wide.";
					break;
				case ENOSPC:
					error_message += "No space left on device.";
					break;
				default:
					error_message += "See errno documentation for details.";
					break;
			}
			throw std::runtime_error(error_message);
		}
	}

	void ingest(const std::string& external_id, const float* vec) {
		if (fd_ < 0) throw std::runtime_error("ingest: writer not initialized");
		if (!vec) throw std::runtime_error("ingest: vec is null");
		if (dim_ == 0) throw std::runtime_error("ingest: dim=0");

		const size_t bytes = dim_ * sizeof(float);
		ssize_t w = ::write(fd_, vec, bytes);
		if (w != (ssize_t)bytes) throw std::runtime_error("ingest: write failed");

		ids_.push_back(external_id);
		++n_;
	}

	void finalize() {
		if (fd_ < 0) throw std::runtime_error("finalize: writer not initialized");
		::fsync(fd_);
		::close(fd_);
		fd_ = -1;
		ghnsw::Index::fsync_dir_of(path_);
	}

	void close() noexcept {
		if (fd_ >= 0) {
			::close(fd_);
			fd_ = -1;
		}
	}

	~VectorIngestWriter() { close(); }

	const std::string& path() const noexcept { return path_; }
	std::size_t dim() const noexcept { return dim_; }
	std::size_t n() const noexcept { return n_; }
	const std::vector<std::string>& ids() const noexcept { return ids_; }

private:
	std::string path_;
	std::size_t dim_ = 0;
	std::size_t n_ = 0;
	int fd_ = -1;
	std::vector<std::string> ids_;
};

struct SearchResult {
	float distance;
	std::string external_id;

	bool operator<(const SearchResult& other) const {
		return distance < other.distance;
	}
};



class VectorEngine {
public:
	enum class PendingMode { None, Build, Insert, Upsert };

	VectorEngine(
		std::string index_path,
		std::size_t dim = 0, 
		float delta_ratio = 0.10,
		ghnsw::Params p = {})
	: base_index_path_(std::move(index_path)), 
	  params_(p)
	{

		if (delta_ratio <= 0.0f || delta_ratio > 0.5f) {
			throw std::invalid_argument("delta_ratio must be in range (0, 0.5]");
		}

		dim_ = dim;
		delta_ratio_ = delta_ratio;

		main_path_ = base_index_path_ + ".main";
		delta_path_ = base_index_path_ + ".delta";
		lock_path_ = base_index_path_ + ".lock";
		p.save_path = main_path_;
		bool iexists = false;
		if (file_exists(main_path_)) {
			main_idx_.reset(new ghnsw::Index(main_path_, params_.ef_search));
			main_size_ = main_idx_->active_size();
			main_mtime_ = file_mtime(main_path_);
			dim_ = main_idx_->dim();
			iexists = true;
		}
		if (dim <= 0 and !iexists) {
			throw std::invalid_argument("dim must be greater than 0");
		}
		if (file_exists(delta_path_)) {
			delta_idx_.reset(new ghnsw::Index(delta_path_, params_.ef_search));
			delta_size_ = delta_idx_->active_size();
			delta_mtime_ = file_mtime(delta_path_);
		}
	}

	~VectorEngine() {
		close();
	}

	std::vector<std::string> search(const float* q, int k, int efs) {
		if (!main_idx_) {
			throw std::runtime_error("search: layout not initialized");
		}

		check_and_reload_if_needed();

		std::vector<SearchResult> all_results;

		auto pairs = main_idx_->query(q, k, efs);
		for (const auto& n : pairs.a) {
			if (!main_idx_->is_deleted(n.id)) {
				all_results.push_back({n.dist, main_idx_->external_id(n.id)});
			}
		}

		if (delta_idx_) {
			auto pairs = delta_idx_->query(q, k, efs);
			for (const auto& n : pairs.a) {
				if (!main_idx_->is_deleted(n.id)) {
					all_results.push_back({n.dist, main_idx_->external_id(n.id)});
				}
			}
		}

		if (all_results.empty()) {
			return {};
		}

		std::sort(all_results.begin(), all_results.end());
		std::vector<std::string> out;
		out.reserve(k);
		for (const auto& res : all_results) {
			out.push_back(res.external_id);
			if ((int)out.size() >= k) break;
		}

		return out;
	}

	std::vector<std::string> map_internal_to_external(const std::vector<uint32_t>& internal) {
		check_and_reload_if_needed();
		ensure_has_any_index();

		std::vector<std::string> out;
		out.reserve(internal.size());

		if (main_idx_) {
			for (uint32_t id : internal) {
				out.push_back(main_idx_->external_id(id));
			}
		} else if (delta_idx_) {
			for (uint32_t id : internal) {
				out.push_back(delta_idx_->external_id(id));
			}
		}

		return out;
	}

	// in-place delete
	std::size_t delete_external_ids(const std::vector<std::string>& ids,
								 std::vector<std::string>* not_found = nullptr)
	{
		CombinedLock lock(lock_path_);
		force_reload_indices();
		ensure_has_any_index();

		std::size_t deleted = 0;
		std::vector<std::string> local_not_found;

		if (main_idx_) {
			deleted += main_idx_->mark_deleted_by_external_ids_and_persist(main_path_, ids, &local_not_found);
		}

		if (delta_idx_) {
			std::vector<std::string> delta_not_found;
			deleted += delta_idx_->mark_deleted_by_external_ids_and_persist(delta_path_, ids, &delta_not_found);

			// merge not_found lists (intersection of both)
			if (main_idx_) {
				std::unordered_set<std::string> main_nf(local_not_found.begin(), local_not_found.end());
				std::vector<std::string> truly_not_found;
				for (const auto& id : delta_not_found) {
					if (main_nf.find(id) != main_nf.end()) {
						truly_not_found.push_back(id);
					}
				}
				local_not_found = truly_not_found;
			} else {
				local_not_found = delta_not_found;
			}
		}

		if (not_found) {
			*not_found = local_not_found;
		}
		// update mtimes after modification
		update_mtimes();
		return deleted;
	}

	// init/ingest/finalize
	void init(const std::string& mode) {
		pending_mode_ = parse_mode(mode);
		pending_vec_path_ = base_index_path_ + ".ingest.tmp." + 
							std::to_string((long long)getpid()) + "." + 
							std::to_string(std::chrono::high_resolution_clock::now()
										   .time_since_epoch().count());
		writer_.init(pending_vec_path_, dim_, /*truncate=*/true);
	}

	void ingest(const std::string& external_id, const float* vec) {
		if (pending_mode_ == PendingMode::None) {
			throw std::runtime_error("ingest: call init() first");
		}
		if (!vec) {
			throw std::runtime_error("ingest: vec is null");
		}

		for (size_t i = 0; i < dim_; ++i) {
			float val = vec[i];
			if (val > 1e16 || 
				val < -1e16) {
				throw std::runtime_error("ingest: vector contains NaN/Inf at index " + std::to_string(i));
			}
		}

		writer_.ingest(external_id, vec);
	}

	void finalize(ghnsw::Params build_params = {}, const bool optimize = false) {
		if (pending_mode_ == PendingMode::None) {
			throw std::runtime_error("finalize: no pending ingest");
		}

		writer_.finalize();
		const std::string tmp_prefix_base = "brinicletmp";
		if (build_params.M <= 0) build_params = params_;

		const size_t new_vectors = writer_.n();

		// acquire exclusive lock
		CombinedLock lock(lock_path_);
		force_reload_indices();
		cleanup_orphaned_temps();

		try {
			if (pending_mode_ == PendingMode::Build) {
				build_from_scratch(build_params, tmp_prefix_base);
			} else if (pending_mode_ == PendingMode::Insert) {
				ensure_has_any_index();

				if (!optimize) {
					absorb_into_delta(build_params, tmp_prefix_base);
				} else {
					size_t projected_delta_size = delta_size_ + new_vectors;
					size_t threshold = (size_t)(main_size_ * delta_ratio_);

					if (projected_delta_size > threshold && main_size_ > 0) {
						merge_delta_and_rebuild(build_params, tmp_prefix_base);
					} else {
						absorb_into_delta(build_params, tmp_prefix_base);
					}
				}
			} else if (pending_mode_ == PendingMode::Upsert) {
				ensure_has_any_index();

				std::vector<std::string> ids_to_delete = writer_.ids();
				auto counts = delete_external_ids_internal(ids_to_delete);
				// you know, cases like deleting all main elements.
				force_reload_indices();
				if ((main_size_ - counts.first) <= 0) {
					build_from_scratch(build_params, tmp_prefix_base);
				} else {
					if (!optimize) {
						absorb_into_delta(build_params, tmp_prefix_base);
					} else {
						size_t projected_delta_size = delta_size_ + new_vectors;
						size_t threshold = (size_t)(main_size_ * delta_ratio_);

						if (projected_delta_size > threshold && main_size_ > 0) {
							merge_delta_and_rebuild(build_params, tmp_prefix_base);
						} else {
							absorb_into_delta(build_params, tmp_prefix_base);
						}
					}
				}
			}
			unlink_noexcept(pending_vec_path_);
		} catch (...) {
			unlink_noexcept(pending_vec_path_);
			throw;
		}

		pending_mode_ = PendingMode::None;
		pending_vec_path_.clear();
	}

	// non-ingest build from existing file
	void build_from_file(const std::string& vectors_path,
		const std::vector<std::string>& external_ids,
		ghnsw::Params build_params = {},
		const std::string& tmp_prefix = "brinicletmp",
		bool delete_vectors_after = true)
	{
		CombinedLock lock(lock_path_);

		if (build_params.M == 0) build_params = params_;

		// build to temp file
		const std::string tmp_main = unique_temp_file(delta_path_, ".new");
		build_params.save_path = tmp_main;

		const std::string tmp = unique_temp_file(tmp_prefix);

		{
			ghnsw::Index builder(vectors_path, external_ids, dim_, build_params, tmp);
			builder.fit();
		}

		cleanup_build_pools(tmp);

		atomic_rename(tmp_main, main_path_);

		force_reload_indices();

		// clear delta segment since we rebuilt everything
		delta_idx_.reset();
		delta_size_ = 0;
		unlink_noexcept(delta_path_);

		if (delete_vectors_after) {
			unlink_noexcept(vectors_path);
		}
	}

	// remove deleted, keep only alive, then rebuild entirely
	void rebuild_compact(ghnsw::Params build_params = {}, const std::string& tmp_prefix_base = "brinicletmp") {
		CombinedLock lock(lock_path_);
		force_reload_indices();
		ensure_has_any_index();

		if (build_params.M == 0) build_params = params_;


		const std::string tmp_main = unique_temp_file(main_path_, ".new");
		build_params.save_path = tmp_main;

		const std::string tmp_prefix = unique_temp_file(tmp_prefix_base);
		const std::string alive_vec_path = unique_temp_file(base_index_path_, ".compact.vec.tmp");

		std::vector<std::string> alive_ids;
		// export alive vectors
		if (main_idx_) {
			main_idx_->export_alive_vectors_to_file(alive_vec_path, alive_ids);
		}

		// append alive vectors from delta
		if (delta_idx_) {
			std::vector<std::string> delta_alive_ids;
			std::string delta_vec_tmp = unique_temp_file(base_index_path_, ".alive.tmp");
			delta_idx_->export_alive_vectors_to_file(delta_vec_tmp, delta_alive_ids);
			append_vectors_to_file(alive_vec_path, delta_vec_tmp, delta_alive_ids);
			alive_ids.insert(alive_ids.end(), delta_alive_ids.begin(), delta_alive_ids.end());
			unlink_noexcept(delta_vec_tmp);
		}

		{
			ghnsw::Index builder(alive_vec_path, alive_ids, dim_, build_params, tmp_prefix);
			builder.fit();
		}

		cleanup_build_pools(tmp_prefix);
		unlink_noexcept(alive_vec_path);

		atomic_rename(tmp_main, main_path_);

		force_reload_indices();

		delta_idx_.reset();
		delta_size_ = 0;
		unlink_noexcept(delta_path_);
	}

	bool has_index() const noexcept { return main_idx_ || delta_idx_; }
	std::size_t dim() const noexcept { return dim_; }

	uint64_t n_deletes_since_build() const {
		const_cast<VectorEngine*>(this)->check_and_reload_if_needed();
		ensure_has_any_index();
		uint64_t total = 0;
		if (main_idx_) total += main_idx_->num_deletes_since_build();
		if (delta_idx_) total += delta_idx_->num_deletes_since_build();
		return total;
	}

	uint64_t n_inserts_since_build() const {
		const_cast<VectorEngine*>(this)->check_and_reload_if_needed();
		ensure_has_any_index();
		uint64_t total = 0;
		if (main_idx_) total += main_idx_->num_inserts_since_build();
		if (delta_idx_) total += delta_idx_->num_inserts_since_build();
		return total;
	}

	std::size_t main_size() const noexcept { return main_size_; }
	std::size_t delta_size() const noexcept { return delta_size_; }
	float delta_ratio() const noexcept { return delta_ratio_; }
	std::size_t delta_threshold() const noexcept {
		return (std::size_t)(main_size_ * delta_ratio_); 
	}

	bool needs_rebuild() noexcept {
		double n = (double)(main_size_ + delta_size_);
		if (n == 0) return false;
		double del_frac = (double)n_deletes_since_build() / n;
		double ins_frac = (double)delta_size() / n;
		if (del_frac >= delta_ratio_) return true;
		if (ins_frac >= delta_ratio_) return true;
		// magic number is not a good practice
		// if ((double)n_deletes_since_build() + (double)delta_size() >= 2e4) return true;
		return false;
	}

	void optimize_graph() {
		if (needs_rebuild()) {
			merge_delta_and_rebuild(params_, "brinicletmp");
		}
	}

	void close() {
		main_idx_.reset();
		delta_idx_.reset();
		main_size_ = 0;
		delta_size_ = 0;
		main_mtime_ = 0;
		delta_mtime_ = 0;
		writer_.close();
		if (!pending_vec_path_.empty()) {
			unlink_noexcept(pending_vec_path_);
			pending_vec_path_.clear();
		}
		pending_mode_ = PendingMode::None;
	}

	void destroy() {
		close();
		CombinedLock lock(lock_path_);
		unlink_noexcept(main_path_);
		unlink_noexcept(delta_path_);
		unlink_noexcept(lock_path_);
		cleanup_orphaned_temps();
	}

private:
	static PendingMode parse_mode(const std::string& m) {
		if (m == "build")  return PendingMode::Build;
		if (m == "insert") return PendingMode::Insert;
		if (m == "upsert") return PendingMode::Upsert;
		throw std::runtime_error("init(mode): mode must be 'build'|'insert'|'upsert'");
	}

	void ensure_has_any_index() const {
		if (!main_idx_ && !delta_idx_) {
			throw std::runtime_error("No index loaded. Build first.");
		}
	}


	void check_and_reload_if_needed() {
		// check modification times
		time_t current_main_mtime = file_exists(main_path_) ? file_mtime(main_path_) : 0;
		time_t current_delta_mtime = file_exists(delta_path_) ? file_mtime(delta_path_) : 0;

		// reload if files changed
		if (current_main_mtime != main_mtime_) {
			if (current_main_mtime > 0) {
				main_idx_.reset(new ghnsw::Index(main_path_, params_.ef_search));
				main_size_ = main_idx_->active_size();
				main_mtime_ = current_main_mtime;
			} else {
				main_idx_.reset();
				main_size_ = 0;
				main_mtime_ = 0;
			}
		}

		if (current_delta_mtime != delta_mtime_) {
			if (current_delta_mtime > 0) {
				delta_idx_.reset(new ghnsw::Index(delta_path_, params_.ef_search));
				delta_size_ = delta_idx_->active_size();
				delta_mtime_ = current_delta_mtime;
			} else {
				delta_idx_.reset();
				delta_size_ = 0;
				delta_mtime_ = 0;
			}
		}
	}

	void force_reload_indices() {
		// used by write operations after acquiring lock
		if (file_exists(main_path_)) {
			main_idx_.reset(new ghnsw::Index(main_path_, params_.ef_search));
			main_size_ = main_idx_->active_size();
			main_mtime_ = file_mtime(main_path_);
		} else {
			main_idx_.reset();
			main_size_ = 0;
			main_mtime_ = 0;
		}
		if (file_exists(delta_path_)) {
			delta_idx_.reset(new ghnsw::Index(delta_path_, params_.ef_search));
			delta_size_ = delta_idx_->active_size();
			delta_mtime_ = file_mtime(delta_path_);
		} else {
			delta_idx_.reset();
			delta_size_ = 0;
			delta_mtime_ = 0;
		}
	}

	void update_mtimes() {
		main_mtime_ = file_exists(main_path_) ? file_mtime(main_path_) : 0;
		delta_mtime_ = file_exists(delta_path_) ? file_mtime(delta_path_) : 0;
	}

	void cleanup_orphaned_temps() {
		// could scan for .tmp files and remove them. tired, perhaps later.
	}

	// internal delete without acquiring lock (already locked by caller)
	std::pair<std::size_t, std::size_t> delete_external_ids_internal(const std::vector<std::string>& ids) {
		std::size_t deleted_main = 0;
		std::size_t deleted_delta = 0;

		if (main_idx_) {
			deleted_main = main_idx_->mark_deleted_by_external_ids_and_persist(main_path_, ids, nullptr);
		}
		if (delta_idx_) {
			deleted_delta = delta_idx_->mark_deleted_by_external_ids_and_persist(delta_path_, ids, nullptr);
		}
		return std::pair<std::size_t, std::size_t>(deleted_main, deleted_delta);
	}

	void build_from_scratch(ghnsw::Params build_params, const std::string& tmp_prefix_base) {
		const std::string tmp_main = main_path_ + ".new";
		build_params.save_path = tmp_main;
		const std::string tmp = unique_temp_file(tmp_prefix_base);

		{
			ghnsw::Index builder(writer_.path(), writer_.ids(), dim_, build_params, tmp);
			builder.fit();
		}

		cleanup_build_pools(tmp);

		atomic_rename(tmp_main, main_path_);

		force_reload_indices();

		delta_idx_.reset();
		delta_size_ = 0;
		unlink_noexcept(delta_path_);
	}

	void absorb_into_delta(ghnsw::Params build_params, const std::string& tmp_prefix_base) {
		std::string combined_vec_path;
		std::vector<std::string> combined_ids;

		if (delta_idx_ && delta_size_ > 0) {
			combined_vec_path = unique_temp_file(delta_path_, ".combined.tmp");
			delta_idx_->export_alive_vectors_to_file(combined_vec_path, combined_ids);

			append_vectors_to_file(combined_vec_path, writer_.path(), writer_.ids());
			combined_ids.insert(combined_ids.end(), writer_.ids().begin(), writer_.ids().end());
		} else {
			combined_vec_path = writer_.path();
			combined_ids = writer_.ids();
		}

		const std::string tmp = unique_temp_file(tmp_prefix_base);
		const std::string tmp_delta = unique_temp_file(delta_path_, ".new");
		build_params.save_path = tmp_delta;

		{
			ghnsw::Index builder(combined_vec_path, combined_ids, dim_, build_params, tmp);
			builder.fit();
		}

		cleanup_build_pools(tmp);

		if (delta_idx_) {
			unlink_noexcept(combined_vec_path);
		}

		atomic_rename(tmp_delta, delta_path_);

		force_reload_indices();
	}

	void merge_delta_and_rebuild(ghnsw::Params build_params, const std::string& tmp_prefix_base) {
		std::string merged_vec_path = unique_temp_file(base_index_path_, ".merged.tmp");
		std::vector<std::string> merged_ids;

		if (main_idx_) {
			main_idx_->export_alive_vectors_to_file(merged_vec_path, merged_ids);
		}

		if (delta_idx_) {
			std::vector<std::string> delta_ids;
			std::string delta_vec_tmp = unique_temp_file(base_index_path_, ".export.tmp");
			delta_idx_->export_alive_vectors_to_file(delta_vec_tmp, delta_ids);

			if (main_idx_) {
				append_vectors_to_file(merged_vec_path, delta_vec_tmp, delta_ids);
			} else {
				merged_vec_path = delta_vec_tmp;
			}

			merged_ids.insert(merged_ids.end(), delta_ids.begin(), delta_ids.end());

			if (main_idx_) {
				unlink_noexcept(delta_vec_tmp);
			}
		}

		append_vectors_to_file(merged_vec_path, writer_.path(), writer_.ids());
		merged_ids.insert(merged_ids.end(), writer_.ids().begin(), writer_.ids().end());

		// const std::string tmp_main = main_path_ + ".new";
		const std::string tmp = unique_temp_file(tmp_prefix_base);
		const std::string tmp_main = unique_temp_file(main_path_, ".new");
		build_params.save_path = tmp_main;

		{
			ghnsw::Index builder(merged_vec_path, merged_ids, dim_, build_params, tmp);
			builder.fit();
		}

		cleanup_build_pools(tmp);
		unlink_noexcept(merged_vec_path);

		atomic_rename(tmp_main, main_path_);

		force_reload_indices();

		delta_idx_.reset();
		delta_size_ = 0;
		unlink_noexcept(delta_path_);
	}

	void append_vectors_to_file(const std::string& dest_path, 
							   const std::string& src_path,
							   const std::vector<std::string>& src_ids) {
		int fd_dest = ::open(dest_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
		if (fd_dest < 0) {
			throw std::runtime_error("append_vectors_to_file: failed to open dest: " + dest_path);
		}

		int fd_src = ::open(src_path.c_str(), O_RDONLY);
		if (fd_src < 0) {
			::close(fd_dest);
			throw std::runtime_error("append_vectors_to_file: failed to open src: " + src_path);
		}

		// char buf[65536];
		char buf[1024];
		ssize_t bytes_read;
		while ((bytes_read = ::read(fd_src, buf, sizeof(buf))) > 0) {
			ssize_t written = ::write(fd_dest, buf, bytes_read);
			if (written != bytes_read) {
				::close(fd_src);
				::close(fd_dest);
				throw std::runtime_error("append_vectors_to_file: write failed");
			}
		}

		::close(fd_src);
		::fsync(fd_dest);
		::close(fd_dest);
		ghnsw::Index::fsync_dir_of(dest_path);
	}

	std::string base_index_path_;
	std::string main_path_;
	std::string delta_path_;
	std::string lock_path_;

	std::size_t dim_ = 0;
	ghnsw::Params params_;

	float delta_ratio_ = 0.10;

	std::unique_ptr<ghnsw::Index> main_idx_;
	std::unique_ptr<ghnsw::Index> delta_idx_;

	std::size_t main_size_ = 0;
	std::size_t delta_size_ = 0;

	// modification times for lazy reload
	time_t main_mtime_ = 0;
	time_t delta_mtime_ = 0;

	// pending ingest
	PendingMode pending_mode_ = PendingMode::None;
	std::string pending_vec_path_;
	VectorIngestWriter writer_;

};
}