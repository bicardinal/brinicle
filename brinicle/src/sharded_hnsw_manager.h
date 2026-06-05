#pragma once

#include "hnsw_manager.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace ghnsw_mgr {

class ShardedVectorEngine {
public:
	ShardedVectorEngine(
		std::string index_path,
		std::size_t n_shards = 1,
		std::size_t dim = 0,
		float delta_ratio = 0.10f,
		ghnsw::Params p = {},
		std::string dist_func = "l2",
		ghnsw::LexicalConfig lexical_cfg = {},
		ghnsw::AutocompleteConfig autocomplete_cfg = {},
		bool auto_reload_manifest = false
	)
	: base_index_path_(std::move(index_path)),
	  manifest_path_(base_index_path_ + ".manifest"),
	  global_lock_path_(base_index_path_ + ".sharded.lock"),
	  n_shards_(n_shards),
	  dim_(dim),
	  params_(p),
	  lexical_cfg_(std::move(lexical_cfg)),
	  autocomplete_cfg_(std::move(autocomplete_cfg)),
	  delta_ratio_(delta_ratio),
	  dist_func_(std::move(dist_func)),
	  auto_reload_manifest_(auto_reload_manifest)
	{
		if (delta_ratio_ <= 0.0f || delta_ratio_ > 0.5f) {
			throw std::invalid_argument("delta_ratio must be in range (0, 0.5]");
		}

		if (file_exists(manifest_path_)) {
			std::ifstream in(manifest_path_);
			if (!in) {
				throw std::runtime_error("failed to open manifest: " + manifest_path_);
			}

			std::unordered_map<std::string, std::string> kv;
			std::string line;

			while (std::getline(in, line)) {
				if (line.empty() || line[0] == '#') continue;

				const std::size_t eq = line.find('=');
				if (eq == std::string::npos) continue;

				std::string key = line.substr(0, eq);
				std::string val = line.substr(eq + 1);

				kv[std::move(key)] = std::move(val);
			}

			auto require = [&](const std::string& key) -> const std::string& {
				auto it = kv.find(key);
				if (it == kv.end()) {
					throw std::runtime_error("manifest missing key: " + key);
				}
				return it->second;
			};

			if (require("magic") != "biscuits_can_fly") {
				throw std::runtime_error("invalid sharded index manifest: " + manifest_path_);
			}

			const std::size_t manifest_n_shards =
				static_cast<std::size_t>(std::stoull(require("n_shards")));

			if (n_shards_ != 0 && n_shards_ != manifest_n_shards) {
				throw std::runtime_error(
					"n_shards mismatch: constructor requested " +
					std::to_string(n_shards_) +
					", manifest has " +
					std::to_string(manifest_n_shards)
				);
			}

			n_shards_ = manifest_n_shards;

			const std::size_t manifest_dim =
				static_cast<std::size_t>(std::stoull(require("dim")));

			if (dim_ != 0 && dim_ != manifest_dim) {
				throw std::runtime_error(
					"dim mismatch: constructor requested " +
					std::to_string(dim_) +
					", manifest has " +
					std::to_string(manifest_dim)
				);
			}

			dim_ = manifest_dim;
			delta_ratio_ = std::stof(require("delta_ratio"));
			dist_func_ = require("dist_func");

			generation_ = static_cast<uint64_t>(std::stoull(require("generation")));

			params_.M = static_cast<std::size_t>(std::stoull(require("M")));
			params_.ef_construction = static_cast<std::size_t>(std::stoull(require("ef_construction")));
			params_.ef_search = static_cast<std::size_t>(std::stoull(require("ef_search")));
			params_.build_n_threads = static_cast<std::size_t>(std::stoull(require("build_n_threads")));
			params_.rng_seed = static_cast<std::size_t>(std::stoull(require("rng_seed")));

			auto get_float = [&](const char* key, float fallback) -> float {
				auto it = kv.find(key);
				if (it == kv.end()) return fallback;
				return std::stof(it->second);
			};

			auto get_bool = [&](const char* key, bool fallback) -> bool {
				auto it = kv.find(key);
				if (it == kv.end()) return fallback;
				return it->second == "1" || it->second == "true";
			};

			lexical_cfg_.build_title_weight = get_float("lexical.build_title_weight", lexical_cfg_.build_title_weight);
			lexical_cfg_.build_attr_weight = get_float("lexical.build_attr_weight", lexical_cfg_.build_attr_weight);
			lexical_cfg_.build_category_weight = get_float("lexical.build_category_weight", lexical_cfg_.build_category_weight);
			lexical_cfg_.build_subcategory_weight = get_float("lexical.build_subcategory_weight", lexical_cfg_.build_subcategory_weight);
			lexical_cfg_.build_vector_weight = get_float("lexical.build_vector_weight", lexical_cfg_.build_vector_weight);
			lexical_cfg_.build_category_penalty = get_float("lexical.build_category_penalty", lexical_cfg_.build_category_penalty);
			lexical_cfg_.build_title_alpha = get_float("lexical.build_title_alpha", lexical_cfg_.build_title_alpha);
			lexical_cfg_.build_title_beta = get_float("lexical.build_title_beta", lexical_cfg_.build_title_beta);

			lexical_cfg_.search_title_weight = get_float("lexical.search_title_weight", lexical_cfg_.search_title_weight);
			lexical_cfg_.search_attr_weight = get_float("lexical.search_attr_weight", lexical_cfg_.search_attr_weight);
			lexical_cfg_.search_category_weight = get_float("lexical.search_category_weight", lexical_cfg_.search_category_weight);
			lexical_cfg_.search_subcategory_weight = get_float("lexical.search_subcategory_weight", lexical_cfg_.search_subcategory_weight);
			lexical_cfg_.search_vector_weight = get_float("lexical.search_vector_weight", lexical_cfg_.search_vector_weight);
			lexical_cfg_.search_category_penalty = get_float("lexical.search_category_penalty", lexical_cfg_.search_category_penalty);
			lexical_cfg_.search_title_alpha = get_float("lexical.search_title_alpha", lexical_cfg_.search_title_alpha);
			lexical_cfg_.search_title_beta = get_float("lexical.search_title_beta", lexical_cfg_.search_title_beta);
			lexical_cfg_.vector_normalized = get_bool("lexical.vector_normalized", lexical_cfg_.vector_normalized);

			autocomplete_cfg_.build_position_decay = get_float("autocomplete.build_position_decay", autocomplete_cfg_.build_position_decay);
			autocomplete_cfg_.build_length_penalty = get_float("autocomplete.build_length_penalty", autocomplete_cfg_.build_length_penalty);
			autocomplete_cfg_.search_position_decay = get_float("autocomplete.search_position_decay", autocomplete_cfg_.search_position_decay);
			autocomplete_cfg_.search_length_penalty = get_float("autocomplete.search_length_penalty", autocomplete_cfg_.search_length_penalty);

			manifest_mtime_ = file_mtime(manifest_path_);
		} else {
			if (n_shards_ == 0) {
				n_shards_ = 1;
			}

			if (n_shards_ == 0) {
				throw std::invalid_argument("n_shards must be greater than 0");
			}

			if (dim_ == 0) {
				throw std::invalid_argument("dim must be greater than 0 when creating a new sharded index");
			}

			CombinedLock lock(global_lock_path_);
			generation_ = 0;
			std::ofstream out(manifest_path_ + ".tmp", std::ios::binary | std::ios::trunc);
			if (!out) {
				throw std::runtime_error("failed to write manifest temp file");
			}

			out << "magic=biscuits_can_fly\n";
			out << "generation=" << generation_ << "\n";
			out << "n_shards=" << n_shards_ << "\n";
			out << "dim=" << dim_ << "\n";
			out << "delta_ratio=" << delta_ratio_ << "\n";
			out << "dist_func=" << dist_func_ << "\n";
			out << "routing=fnv1a_external_id_mod_n_shards\n";
			out << "M=" << params_.M << "\n";
			out << "ef_construction=" << params_.ef_construction << "\n";
			out << "ef_search=" << params_.ef_search << "\n";
			out << "build_n_threads=" << params_.build_n_threads << "\n";
			out << "rng_seed=" << params_.rng_seed << "\n";

			out << "lexical.build_title_weight=" << lexical_cfg_.build_title_weight << "\n";
			out << "lexical.build_attr_weight=" << lexical_cfg_.build_attr_weight << "\n";
			out << "lexical.build_category_weight=" << lexical_cfg_.build_category_weight << "\n";
			out << "lexical.build_subcategory_weight=" << lexical_cfg_.build_subcategory_weight << "\n";
			out << "lexical.build_vector_weight=" << lexical_cfg_.build_vector_weight << "\n";
			out << "lexical.build_category_penalty=" << lexical_cfg_.build_category_penalty << "\n";
			out << "lexical.build_title_alpha=" << lexical_cfg_.build_title_alpha << "\n";
			out << "lexical.build_title_beta=" << lexical_cfg_.build_title_beta << "\n";

			out << "lexical.search_title_weight=" << lexical_cfg_.search_title_weight << "\n";
			out << "lexical.search_attr_weight=" << lexical_cfg_.search_attr_weight << "\n";
			out << "lexical.search_category_weight=" << lexical_cfg_.search_category_weight << "\n";
			out << "lexical.search_subcategory_weight=" << lexical_cfg_.search_subcategory_weight << "\n";
			out << "lexical.search_vector_weight=" << lexical_cfg_.search_vector_weight << "\n";
			out << "lexical.search_category_penalty=" << lexical_cfg_.search_category_penalty << "\n";
			out << "lexical.search_title_alpha=" << lexical_cfg_.search_title_alpha << "\n";
			out << "lexical.search_title_beta=" << lexical_cfg_.search_title_beta << "\n";
			out << "lexical.vector_normalized=" << (lexical_cfg_.vector_normalized ? 1 : 0) << "\n";

			out << "autocomplete.build_position_decay=" << autocomplete_cfg_.build_position_decay << "\n";
			out << "autocomplete.build_length_penalty=" << autocomplete_cfg_.build_length_penalty << "\n";
			out << "autocomplete.search_position_decay=" << autocomplete_cfg_.search_position_decay << "\n";
			out << "autocomplete.search_length_penalty=" << autocomplete_cfg_.search_length_penalty << "\n";

			out.close();

			atomic_rename(manifest_path_ + ".tmp", manifest_path_);
			manifest_mtime_ = file_mtime(manifest_path_);
		}

		if (n_shards_ == 0) {
			throw std::invalid_argument("n_shards must be greater than 0");
		}

		shards_.reserve(n_shards_);

		for (std::size_t i = 0; i < n_shards_; ++i) {
			std::ostringstream shard_path;
			shard_path << base_index_path_ << ".shard." << i;

			shards_.emplace_back(new VectorEngine(
				shard_path.str(),
				dim_,
				delta_ratio_,
				params_,
				dist_func_,
				lexical_cfg_,
				autocomplete_cfg_
			));
		}

		shard_pending_.assign(n_shards_, false);
	}

	~ShardedVectorEngine() {
		close();
	}

	void init(const std::string& mode) {
		if (mode != "build" && mode != "insert" && mode != "upsert") {
			throw std::runtime_error("init(mode): mode must be 'build'|'insert'|'upsert'");
		}

		if (pending_) {
			throw std::runtime_error("init: ingest is already pending");
		}

		pending_ = true;
		pending_mode_ = mode;
		shard_pending_.assign(n_shards_, false);
	}

	void ingest(const std::string& external_id, const float* vec) {
		if (!pending_) {
			throw std::runtime_error("ingest: call init() first");
		}

		if (!vec) {
			throw std::runtime_error("ingest: vec is null");
		}

		const std::size_t shard_id =
			static_cast<std::size_t>(fnv1a64(external_id) % n_shards_);

		if (!shard_pending_[shard_id]) {
			shards_[shard_id]->init(pending_mode_);
			shard_pending_[shard_id] = true;
		}

		shards_[shard_id]->ingest(external_id, vec);
	}

	void finalize(ghnsw::Params build_params = {}, const bool optimize = false) {
		if (!pending_) {
			throw std::runtime_error("finalize: no pending ingest");
		}

		CombinedLock lock(global_lock_path_);

		try {
			for (std::size_t i = 0; i < n_shards_; ++i) {
				if (shard_pending_[i]) {
					shards_[i]->finalize(build_params, optimize);
				} else if (pending_mode_ == "build") {
					shards_[i]->destroy();
				}
			}

			++generation_;

			std::ofstream out(manifest_path_ + ".tmp", std::ios::binary | std::ios::trunc);
			if (!out) {
				throw std::runtime_error("failed to write manifest temp file");
			}

			out << "magic=biscuits_can_fly\n";
			out << "generation=" << generation_ << "\n";
			out << "n_shards=" << n_shards_ << "\n";
			out << "dim=" << dim_ << "\n";
			out << "delta_ratio=" << delta_ratio_ << "\n";
			out << "dist_func=" << dist_func_ << "\n";
			out << "routing=fnv1a_external_id_mod_n_shards\n";
			out << "M=" << params_.M << "\n";
			out << "ef_construction=" << params_.ef_construction << "\n";
			out << "ef_search=" << params_.ef_search << "\n";
			out << "build_n_threads=" << params_.build_n_threads << "\n";
			out << "rng_seed=" << params_.rng_seed << "\n";

			out << "lexical.build_title_weight=" << lexical_cfg_.build_title_weight << "\n";
			out << "lexical.build_attr_weight=" << lexical_cfg_.build_attr_weight << "\n";
			out << "lexical.build_category_weight=" << lexical_cfg_.build_category_weight << "\n";
			out << "lexical.build_subcategory_weight=" << lexical_cfg_.build_subcategory_weight << "\n";
			out << "lexical.build_vector_weight=" << lexical_cfg_.build_vector_weight << "\n";
			out << "lexical.build_category_penalty=" << lexical_cfg_.build_category_penalty << "\n";
			out << "lexical.build_title_alpha=" << lexical_cfg_.build_title_alpha << "\n";
			out << "lexical.build_title_beta=" << lexical_cfg_.build_title_beta << "\n";

			out << "lexical.search_title_weight=" << lexical_cfg_.search_title_weight << "\n";
			out << "lexical.search_attr_weight=" << lexical_cfg_.search_attr_weight << "\n";
			out << "lexical.search_category_weight=" << lexical_cfg_.search_category_weight << "\n";
			out << "lexical.search_subcategory_weight=" << lexical_cfg_.search_subcategory_weight << "\n";
			out << "lexical.search_vector_weight=" << lexical_cfg_.search_vector_weight << "\n";
			out << "lexical.search_category_penalty=" << lexical_cfg_.search_category_penalty << "\n";
			out << "lexical.search_title_alpha=" << lexical_cfg_.search_title_alpha << "\n";
			out << "lexical.search_title_beta=" << lexical_cfg_.search_title_beta << "\n";
			out << "lexical.vector_normalized=" << (lexical_cfg_.vector_normalized ? 1 : 0) << "\n";

			out << "autocomplete.build_position_decay=" << autocomplete_cfg_.build_position_decay << "\n";
			out << "autocomplete.build_length_penalty=" << autocomplete_cfg_.build_length_penalty << "\n";
			out << "autocomplete.search_position_decay=" << autocomplete_cfg_.search_position_decay << "\n";
			out << "autocomplete.search_length_penalty=" << autocomplete_cfg_.search_length_penalty << "\n";

			out.close();

			atomic_rename(manifest_path_ + ".tmp", manifest_path_);
			manifest_mtime_ = file_mtime(manifest_path_);

			pending_ = false;
			pending_mode_.clear();
			std::fill(shard_pending_.begin(), shard_pending_.end(), false);
		} catch (...) {
			pending_ = false;
			pending_mode_.clear();
			std::fill(shard_pending_.begin(), shard_pending_.end(), false);
			throw;
		}
	}

	std::vector<std::pair<std::string, float>> search_with_distance(
		const float* q,
		const int k,
		const int efs,
		float threshold = std::numeric_limits<float>::infinity(),
		int n_jobs = 1
	) {
		if (!q || k <= 0) {
			return {};
		}

		struct Candidate {
			float distance;
			uint32_t internal_id;
			uint32_t shard_id;
			uint8_t segment; // 0 = main, 1 = delta

			bool operator<(const Candidate& other) const noexcept {
				return distance < other.distance;
			}
		};

		const bool use_threshold = std::isfinite(threshold);
		const std::size_t per_segment_cap = static_cast<std::size_t>(std::max(efs, k));
		const std::size_t max_per_shard = per_segment_cap * static_cast<std::size_t>(2);
		const std::size_t max_candidates = n_shards_ * max_per_shard;

		std::vector<Candidate> candidate_slots(max_candidates);
		std::vector<uint32_t> counts(n_shards_, 0);

#ifdef _OPENMP
		const int prev_threads = omp_get_max_threads();
		if (n_jobs > 0) {
			omp_set_num_threads(n_jobs);
		}
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_jobs != 1)
#endif
		for (std::ptrdiff_t si = 0; si < static_cast<std::ptrdiff_t>(n_shards_); ++si) {
			const std::size_t shard_id = static_cast<std::size_t>(si);
			VectorEngine& shard = *shards_[shard_id];

			Candidate* dst = candidate_slots.data() + shard_id * max_per_shard;
			uint32_t written = 0;

			if (shard.main_idx_) {
				auto pairs = shard.main_idx_->query(q, k, efs);
				const bool has_deletes = shard.main_idx_->num_deletes_since_build() > 0;

				if (!has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						dst[written++] = Candidate{
							p.first,
							static_cast<uint32_t>(p.second),
							static_cast<uint32_t>(shard_id),
							0
						};
					}
				} else if (!has_deletes && use_threshold) {
					for (const auto& p : pairs) {
						if (p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								static_cast<uint32_t>(p.second),
								static_cast<uint32_t>(shard_id),
								0
							};
						}
					}
				} else if (has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.main_idx_->is_deleted(internal_id)) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								0
							};
						}
					}
				} else {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.main_idx_->is_deleted(internal_id) && p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								0
							};
						}
					}
				}
			}

			if (shard.delta_idx_) {
				auto pairs = shard.delta_idx_->query(q, k, efs);
				const bool has_deletes = shard.delta_idx_->num_deletes_since_build() > 0;

				if (!has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						dst[written++] = Candidate{
							p.first,
							static_cast<uint32_t>(p.second),
							static_cast<uint32_t>(shard_id),
							1
						};
					}
				} else if (!has_deletes && use_threshold) {
					for (const auto& p : pairs) {
						if (p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								static_cast<uint32_t>(p.second),
								static_cast<uint32_t>(shard_id),
								1
							};
						}
					}
				} else if (has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.delta_idx_->is_deleted(internal_id)) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								1
							};
						}
					}
				} else {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.delta_idx_->is_deleted(internal_id) && p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								1
							};
						}
					}
				}
			}

			counts[shard_id] = written;
		}

#ifdef _OPENMP
		if (n_jobs > 0) {
			omp_set_num_threads(prev_threads);
		}
#endif

		std::size_t total = 0;
		for (uint32_t n : counts) {
			total += static_cast<std::size_t>(n);
		}

		if (total == 0) {
			return {};
		}

		std::vector<Candidate> candidates;
		candidates.reserve(total);

		for (std::size_t shard_id = 0; shard_id < n_shards_; ++shard_id) {
			const uint32_t n = counts[shard_id];

			if (n == 0) {
				continue;
			}

			Candidate* first = candidate_slots.data() + shard_id * max_per_shard;
			candidates.insert(
				candidates.end(),
				std::make_move_iterator(first),
				std::make_move_iterator(first + n)
			);
		}

		const std::size_t limit = std::min<std::size_t>(
			static_cast<std::size_t>(k),
			candidates.size()
		);

		if (candidates.size() > limit) {
			std::nth_element(
				candidates.begin(),
				candidates.begin() + static_cast<std::ptrdiff_t>(limit),
				candidates.end()
			);

			candidates.resize(limit);
		}

		std::sort(candidates.begin(), candidates.end());

		std::vector<std::pair<std::string, float>> out;
		out.reserve(limit);

		for (const Candidate& c : candidates) {
			VectorEngine& shard = *shards_[static_cast<std::size_t>(c.shard_id)];

			if (c.segment == 0) {
				out.emplace_back(
					shard.main_idx_->external_id(c.internal_id),
					c.distance
				);
			} else {
				out.emplace_back(
					shard.delta_idx_->external_id(c.internal_id),
					c.distance
				);
			}
		}

		return out;
	}

	std::vector<std::string> search(
		const float* q,
		const int k,
		const int efs,
		float threshold = std::numeric_limits<float>::infinity(),
		int n_jobs = 1
	) {
		if (!q || k <= 0) {
			return {};
		}

		struct Candidate {
			float distance;
			uint32_t internal_id;
			uint32_t shard_id;
			uint8_t segment; // 0 = main, 1 = delta

			bool operator<(const Candidate& other) const noexcept {
				return distance < other.distance;
			}
		};

		const bool use_threshold = std::isfinite(threshold);
		const std::size_t per_segment_cap = static_cast<std::size_t>(std::max(efs, k));
		const std::size_t max_per_shard = per_segment_cap * static_cast<std::size_t>(2);
		const std::size_t max_candidates = n_shards_ * max_per_shard;

		std::vector<Candidate> candidate_slots(max_candidates);
		std::vector<uint32_t> counts(n_shards_, 0);

#ifdef _OPENMP
		const int prev_threads = omp_get_max_threads();
		if (n_jobs > 0) {
			omp_set_num_threads(n_jobs);
		}
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_jobs != 1)
#endif
		for (std::ptrdiff_t si = 0; si < static_cast<std::ptrdiff_t>(n_shards_); ++si) {
			const std::size_t shard_id = static_cast<std::size_t>(si);
			VectorEngine& shard = *shards_[shard_id];

			Candidate* dst = candidate_slots.data() + shard_id * max_per_shard;
			uint32_t written = 0;

			if (shard.main_idx_) {
				auto pairs = shard.main_idx_->query(q, k, efs);
				const bool has_deletes = shard.main_idx_->num_deletes_since_build() > 0;

				if (!has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						dst[written++] = Candidate{
							p.first,
							static_cast<uint32_t>(p.second),
							static_cast<uint32_t>(shard_id),
							0
						};
					}
				} else if (!has_deletes && use_threshold) {
					for (const auto& p : pairs) {
						if (p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								static_cast<uint32_t>(p.second),
								static_cast<uint32_t>(shard_id),
								0
							};
						}
					}
				} else if (has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.main_idx_->is_deleted(internal_id)) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								0
							};
						}
					}
				} else {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.main_idx_->is_deleted(internal_id) && p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								0
							};
						}
					}
				}
			}

			if (shard.delta_idx_) {
				auto pairs = shard.delta_idx_->query(q, k, efs);
				const bool has_deletes = shard.delta_idx_->num_deletes_since_build() > 0;

				if (!has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						dst[written++] = Candidate{
							p.first,
							static_cast<uint32_t>(p.second),
							static_cast<uint32_t>(shard_id),
							1
						};
					}
				} else if (!has_deletes && use_threshold) {
					for (const auto& p : pairs) {
						if (p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								static_cast<uint32_t>(p.second),
								static_cast<uint32_t>(shard_id),
								1
							};
						}
					}
				} else if (has_deletes && !use_threshold) {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.delta_idx_->is_deleted(internal_id)) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								1
							};
						}
					}
				} else {
					for (const auto& p : pairs) {
						const uint32_t internal_id = static_cast<uint32_t>(p.second);

						if (!shard.delta_idx_->is_deleted(internal_id) && p.first < threshold) {
							dst[written++] = Candidate{
								p.first,
								internal_id,
								static_cast<uint32_t>(shard_id),
								1
							};
						}
					}
				}
			}

			counts[shard_id] = written;
		}

#ifdef _OPENMP
		if (n_jobs > 0) {
			omp_set_num_threads(prev_threads);
		}
#endif

		std::size_t total = 0;
		for (uint32_t n : counts) {
			total += static_cast<std::size_t>(n);
		}

		if (total == 0) {
			return {};
		}

		std::vector<Candidate> candidates;
		candidates.reserve(total);

		for (std::size_t shard_id = 0; shard_id < n_shards_; ++shard_id) {
			const uint32_t n = counts[shard_id];

			if (n == 0) {
				continue;
			}

			Candidate* first = candidate_slots.data() + shard_id * max_per_shard;
			candidates.insert(
				candidates.end(),
				std::make_move_iterator(first),
				std::make_move_iterator(first + n)
			);
		}

		const std::size_t limit = std::min<std::size_t>(
			static_cast<std::size_t>(k),
			candidates.size()
		);

		if (candidates.size() > limit) {
			std::nth_element(
				candidates.begin(),
				candidates.begin() + static_cast<std::ptrdiff_t>(limit),
				candidates.end()
			);

			candidates.resize(limit);
		}

		std::sort(candidates.begin(), candidates.end());

		std::vector<std::string> out;
		out.reserve(limit);

		for (const Candidate& c : candidates) {
			VectorEngine& shard = *shards_[static_cast<std::size_t>(c.shard_id)];

			if (c.segment == 0) {
				out.emplace_back(shard.main_idx_->external_id(c.internal_id));
			} else {
				out.emplace_back(shard.delta_idx_->external_id(c.internal_id));
			}
		}

		return out;
	}

	std::vector<std::vector<std::string>> search_batch(
		const float* Q,
		std::size_t nq,
		const int k,
		const int efs,
		float threshold = std::numeric_limits<float>::infinity(),
		int n_jobs = 1
	) {
		std::vector<std::vector<std::string>> out(nq);

		if (!Q || nq == 0 || k <= 0) {
			return out;
		}

		struct Candidate {
			float distance;
			uint32_t internal_id;
			uint32_t shard_id;
			uint8_t segment; // 0 = main, 1 = delta

			bool operator<(const Candidate& other) const noexcept {
				return distance < other.distance;
			}
		};

		const bool use_threshold = std::isfinite(threshold);
		const std::size_t max_candidates =
		    n_shards_ * efs * static_cast<std::size_t>(2);

		auto process_query = [&](std::size_t qi, std::vector<Candidate>& candidates) {
			const float* q = Q + qi * dim_;

			candidates.clear();

			for (std::size_t shard_id = 0; shard_id < n_shards_; ++shard_id) {
				VectorEngine& shard = *shards_[shard_id];

				const uint32_t sid = static_cast<uint32_t>(shard_id);

				if (shard.main_idx_) {
					auto pairs = shard.main_idx_->query(q, k, efs);
					const bool has_deletes =
						shard.main_idx_->num_deletes_since_build() > 0;

					if (!has_deletes && !use_threshold) {
						for (const auto& p : pairs) {
							candidates.push_back(Candidate{
								p.first,
								static_cast<uint32_t>(p.second),
								sid,
								0
							});
						}
					} else if (!has_deletes && use_threshold) {
						for (const auto& p : pairs) {
							if (p.first < threshold) {
								candidates.push_back(Candidate{
									p.first,
									static_cast<uint32_t>(p.second),
									sid,
									0
								});
							}
						}
					} else if (has_deletes && !use_threshold) {
						for (const auto& p : pairs) {
							const uint32_t internal_id =
								static_cast<uint32_t>(p.second);

							if (!shard.main_idx_->is_deleted(internal_id)) {
								candidates.push_back(Candidate{
									p.first,
									internal_id,
									sid,
									0
								});
							}
						}
					} else {
						for (const auto& p : pairs) {
							const uint32_t internal_id =
								static_cast<uint32_t>(p.second);

							if (!shard.main_idx_->is_deleted(internal_id) &&
								p.first < threshold) {
								candidates.push_back(Candidate{
									p.first,
									internal_id,
									sid,
									0
								});
							}
						}
					}
				}

				if (shard.delta_idx_) {
					auto pairs = shard.delta_idx_->query(q, k, efs);
					const bool has_deletes =
						shard.delta_idx_->num_deletes_since_build() > 0;

					if (!has_deletes && !use_threshold) {
						for (const auto& p : pairs) {
							candidates.push_back(Candidate{
								p.first,
								static_cast<uint32_t>(p.second),
								sid,
								1
							});
						}
					} else if (!has_deletes && use_threshold) {
						for (const auto& p : pairs) {
							if (p.first < threshold) {
								candidates.push_back(Candidate{
									p.first,
									static_cast<uint32_t>(p.second),
									sid,
									1
								});
							}
						}
					} else if (has_deletes && !use_threshold) {
						for (const auto& p : pairs) {
							const uint32_t internal_id =
								static_cast<uint32_t>(p.second);

							if (!shard.delta_idx_->is_deleted(internal_id)) {
								candidates.push_back(Candidate{
									p.first,
									internal_id,
									sid,
									1
								});
							}
						}
					} else {
						for (const auto& p : pairs) {
							const uint32_t internal_id =
								static_cast<uint32_t>(p.second);

							if (!shard.delta_idx_->is_deleted(internal_id) &&
								p.first < threshold) {
								candidates.push_back(Candidate{
									p.first,
									internal_id,
									sid,
									1
								});
							}
						}
					}
				}
			}

			if (candidates.empty()) {
				return;
			}

			const std::size_t limit = std::min<std::size_t>(
				static_cast<std::size_t>(k),
				candidates.size()
			);

			if (candidates.size() > limit) {
				std::nth_element(
					candidates.begin(),
					candidates.begin() + static_cast<std::ptrdiff_t>(limit),
					candidates.end()
				);

				candidates.resize(limit);
			}

			std::sort(candidates.begin(), candidates.end());

			std::vector<std::string>& row = out[qi];
			row.clear();
			row.reserve(limit);

			for (const Candidate& c : candidates) {
				VectorEngine& shard = *shards_[static_cast<std::size_t>(c.shard_id)];

				if (c.segment == 0) {
					row.push_back(shard.main_idx_->external_id(c.internal_id));
				} else {
					row.push_back(shard.delta_idx_->external_id(c.internal_id));
				}
			}
		};

#ifdef _OPENMP
		const int prev_threads = omp_get_max_threads();

		if (n_jobs > 0) {
			omp_set_num_threads(n_jobs);
		}

#pragma omp parallel if(n_jobs != 1)
		{
			std::vector<Candidate> candidates;
			candidates.reserve(max_candidates);

#pragma omp for schedule(dynamic, 1)
			for (std::ptrdiff_t qi = 0; qi < static_cast<std::ptrdiff_t>(nq); ++qi) {
				process_query(static_cast<std::size_t>(qi), candidates);
			}
		}

		if (n_jobs > 0) {
			omp_set_num_threads(prev_threads);
		}
#else
		std::vector<Candidate> candidates;
		candidates.reserve(max_candidates);

		for (std::size_t qi = 0; qi < nq; ++qi) {
			process_query(qi, candidates);
		}
#endif
		return out;
	}

	std::size_t delete_external_ids(
		const std::vector<std::string>& ids,
		std::vector<std::string>* not_found = nullptr
	) {
		if (ids.empty()) {
			if (not_found) not_found->clear();
			return 0;
		}

		CombinedLock lock(global_lock_path_);

		std::vector<std::vector<std::string>> grouped(n_shards_);

		for (const auto& id : ids) {
			const std::size_t shard_id =
				static_cast<std::size_t>(fnv1a64(id) % n_shards_);

			grouped[shard_id].push_back(id);
		}

		std::size_t deleted = 0;
		std::vector<std::string> nf;

		for (std::size_t i = 0; i < n_shards_; ++i) {
			if (grouped[i].empty()) continue;

			std::vector<std::string> shard_nf;

			deleted += shards_[i]->delete_external_ids(
				grouped[i],
				not_found ? &shard_nf : nullptr
			);

			if (not_found) {
				nf.insert(
					nf.end(),
					std::make_move_iterator(shard_nf.begin()),
					std::make_move_iterator(shard_nf.end())
				);
			}
		}

		if (deleted > 0) {
			++generation_;

			std::ofstream out(manifest_path_ + ".tmp", std::ios::binary | std::ios::trunc);
			if (!out) {
				throw std::runtime_error("failed to write manifest temp file after delete");
			}

			out << "magic=biscuits_can_fly\n";
			out << "generation=" << generation_ << "\n";
			out << "n_shards=" << n_shards_ << "\n";
			out << "dim=" << dim_ << "\n";
			out << "delta_ratio=" << delta_ratio_ << "\n";
			out << "dist_func=" << dist_func_ << "\n";
			out << "routing=fnv1a_external_id_mod_n_shards\n";
			out << "M=" << params_.M << "\n";
			out << "ef_construction=" << params_.ef_construction << "\n";
			out << "ef_search=" << params_.ef_search << "\n";
			out << "build_n_threads=" << params_.build_n_threads << "\n";
			out << "rng_seed=" << params_.rng_seed << "\n";
			out.close();

			atomic_rename(manifest_path_ + ".tmp", manifest_path_);
			manifest_mtime_ = file_mtime(manifest_path_);
		}

		if (not_found) {
			*not_found = std::move(nf);
		}

		return deleted;
	}

	void rebuild_compact(ghnsw::Params build_params = {}) {
		CombinedLock lock(global_lock_path_);

		for (auto& shard : shards_) {
			if (shard->has_index()) {
				shard->rebuild_compact(build_params);
			}
		}

		++generation_;

		std::ofstream out(manifest_path_ + ".tmp", std::ios::binary | std::ios::trunc);
		if (!out) {
			throw std::runtime_error("failed to write manifest temp file after rebuild_compact");
		}

		out << "magic=biscuits_can_fly\n";
		out << "generation=" << generation_ << "\n";
		out << "n_shards=" << n_shards_ << "\n";
		out << "dim=" << dim_ << "\n";
		out << "delta_ratio=" << delta_ratio_ << "\n";
		out << "dist_func=" << dist_func_ << "\n";
		out << "routing=fnv1a_external_id_mod_n_shards\n";
		out << "M=" << params_.M << "\n";
		out << "ef_construction=" << params_.ef_construction << "\n";
		out << "ef_search=" << params_.ef_search << "\n";
		out << "build_n_threads=" << params_.build_n_threads << "\n";
		out << "rng_seed=" << params_.rng_seed << "\n";
		out.close();

		atomic_rename(manifest_path_ + ".tmp", manifest_path_);
		manifest_mtime_ = file_mtime(manifest_path_);
	}

	bool needs_rebuild() noexcept {
		for (auto& shard : shards_) {
			if (shard->has_index() && shard->needs_rebuild()) {
				return true;
			}
		}
		return false;
	}

	void optimize_graph() {
		CombinedLock lock(global_lock_path_);

		bool touched = false;

		for (auto& shard : shards_) {
			if (shard->has_index() && shard->needs_rebuild()) {
				shard->optimize_graph();
				touched = true;
			}
		}

		if (touched) {
			++generation_;

			std::ofstream out(manifest_path_ + ".tmp", std::ios::binary | std::ios::trunc);
			if (!out) {
				throw std::runtime_error("failed to write manifest temp file after optimize_graph");
			}

			out << "magic=biscuits_can_fly\n";
			out << "generation=" << generation_ << "\n";
			out << "n_shards=" << n_shards_ << "\n";
			out << "dim=" << dim_ << "\n";
			out << "delta_ratio=" << delta_ratio_ << "\n";
			out << "dist_func=" << dist_func_ << "\n";
			out << "routing=fnv1a_external_id_mod_n_shards\n";
			out << "M=" << params_.M << "\n";
			out << "ef_construction=" << params_.ef_construction << "\n";
			out << "ef_search=" << params_.ef_search << "\n";
			out << "build_n_threads=" << params_.build_n_threads << "\n";
			out << "rng_seed=" << params_.rng_seed << "\n";
			out.close();

			atomic_rename(manifest_path_ + ".tmp", manifest_path_);
			manifest_mtime_ = file_mtime(manifest_path_);
		}
	}

	void close() {
		for (auto& shard : shards_) {
			if (shard) {
				shard->close();
			}
		}

		pending_ = false;
		pending_mode_.clear();

		if (!shard_pending_.empty()) {
			std::fill(shard_pending_.begin(), shard_pending_.end(), false);
		}
	}

	void destroy() {
		close();

		CombinedLock lock(global_lock_path_);

		for (auto& shard : shards_) {
			if (shard) {
				shard->destroy();
			}
		}

		unlink_noexcept(manifest_path_);
		unlink_noexcept(global_lock_path_);
	}

	bool has_index() const noexcept {
		for (const auto& shard : shards_) {
			if (shard && shard->has_index()) {
				return true;
			}
		}
		return false;
	}

	std::size_t dim() const noexcept {
		return dim_;
	}

	std::size_t n_shards() const noexcept {
		return n_shards_;
	}

	uint64_t generation() const noexcept {
		return generation_;
	}

	std::size_t main_size() const noexcept {
		std::size_t total = 0;
		for (const auto& shard : shards_) {
			if (shard) total += shard->main_size();
		}
		return total;
	}

	std::size_t delta_size() const noexcept {
		std::size_t total = 0;
		for (const auto& shard : shards_) {
			if (shard) total += shard->delta_size();
		}
		return total;
	}

private:
	static uint64_t fnv1a64(const std::string& s) noexcept {
		uint64_t h = 14695981039346656037ull;

		for (unsigned char c : s) {
			h ^= static_cast<uint64_t>(c);
			h *= 1099511628211ull;
		}

		return h;
	}

	std::string base_index_path_;
	std::string manifest_path_;
	std::string global_lock_path_;

	std::size_t n_shards_ = 1;
	std::size_t dim_ = 0;

	ghnsw::Params params_;
	ghnsw::LexicalConfig lexical_cfg_;
	ghnsw::AutocompleteConfig autocomplete_cfg_;

	float delta_ratio_ = 0.10f;
	std::string dist_func_;

	uint64_t generation_ = 0;
	time_t manifest_mtime_ = 0;
	bool auto_reload_manifest_ = false;

	std::vector<std::unique_ptr<VectorEngine>> shards_;

	bool pending_ = false;
	std::string pending_mode_;
	std::vector<bool> shard_pending_;
};

} // namespace ghnsw_mgr