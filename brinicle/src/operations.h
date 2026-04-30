
#pragma once
#ifndef L2_HSUM_UTILS_H
#define L2_HSUM_UTILS_H

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <algorithm>

namespace ops {


#if (defined(__clang__) && __clang_major__ >= 16) || (defined(__GNUC__) && __GNUC__ >= 14)
#  if defined(__AVX512F__) && defined(__AVX512VL__)
#    define HAS_AVX512_REDUCE 1
#  endif
#endif

static inline float hsum256_ps(__m256 v) noexcept
{
#if defined(HAS_AVX512_REDUCE)
	return _mm256_reduce_add_ps(v);
#else
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 sum = _mm_add_ps(lo, hi);

	__m128 shuf = _mm_movehdup_ps(sum);
	sum        = _mm_add_ps(sum, shuf);
	shuf       = _mm_movehl_ps(shuf, sum);
	sum        = _mm_add_ss(sum, shuf);

	return _mm_cvtss_f32(sum);
#endif
}


// 512‑bit fallback
static inline float hsum512_ps(__m512 v) noexcept
{
#if defined(__AVX512F__)
#  if defined(__clang__) && __clang_major__ >= 16
	  return _mm512_reduce_add_ps(v);
#  elif defined(__GNUC__) && __GNUC__ >= 14
	  return _mm512_reduce_add_ps(v);
#  else
	  __m256 lo = _mm512_castps512_ps256(v);            // lower 256
	  __m256 hi = _mm512_extractf32x8_ps(v, 1);         // upper 256
	  return hsum256_ps(_mm256_add_ps(lo, hi));
#  endif
#else
	return 0.0f;
#endif
}

#endif // L2_HSUM_UTILS_H



inline float l2_sqr(const float* a, const float* b, std::size_t dim) noexcept {
#if defined(__AVX512F__)
	__m512 acc0 = _mm512_setzero_ps();
	__m512 acc1 = _mm512_setzero_ps();

	std::size_t i = 0;
	for (; i + 32 <= dim; i += 32) {
		__m512 va0 = _mm512_loadu_ps(a + i);
		__m512 vb0 = _mm512_loadu_ps(b + i);
		__m512 diff0 = _mm512_sub_ps(va0, vb0);
		acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

		__m512 va1 = _mm512_loadu_ps(a + i + 16);
		__m512 vb1 = _mm512_loadu_ps(b + i + 16);
		__m512 diff1 = _mm512_sub_ps(va1, vb1);
		acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);
	}

	__m512 acc = _mm512_add_ps(acc0, acc1);

	/* handle 16‑float chunk if (dim % 32) ≥ 16 */
	if (i + 16 <= dim) {
		__m512 va = _mm512_loadu_ps(a + i);
		__m512 vb = _mm512_loadu_ps(b + i);
		__m512 diff = _mm512_sub_ps(va, vb);
		acc = _mm512_fmadd_ps(diff, diff, acc);
		i += 16;
	}

	float sum = hsum512_ps(acc);
#elif defined(__AVX2__) && defined(__FMA__)
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();

	std::size_t i = 0;
	for (; i + 16 <= dim; i += 16) {
		__m256 va0 = _mm256_loadu_ps(a + i);
		__m256 vb0 = _mm256_loadu_ps(b + i);
		__m256 diff0 = _mm256_sub_ps(va0, vb0);
		acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

		__m256 va1 = _mm256_loadu_ps(a + i + 8);
		__m256 vb1 = _mm256_loadu_ps(b + i + 8);
		__m256 diff1 = _mm256_sub_ps(va1, vb1);
		acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
	}

	__m256 acc = _mm256_add_ps(acc0, acc1);

	/* handle 8‑float chunk if (dim % 16) ≥ 8 */
	if (i + 8 <= dim) {
		__m256 va = _mm256_loadu_ps(a + i);
		__m256 vb = _mm256_loadu_ps(b + i);
		__m256 diff = _mm256_sub_ps(va, vb);
		acc = _mm256_fmadd_ps(diff, diff, acc);
		i += 8;
	}

	float sum = hsum256_ps(acc);   // GCC/Clang idiom
#elif defined(__SSE2__)
	__m128 acc0 = _mm_setzero_ps();
	__m128 acc1 = _mm_setzero_ps();

	std::size_t i = 0;
	for (; i + 8 <= dim; i += 8) {
		__m128 va0 = _mm_loadu_ps(a + i);
		__m128 vb0 = _mm_loadu_ps(b + i);
		__m128 diff0 = _mm_sub_ps(va0, vb0);
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(diff0, diff0));

		__m128 va1 = _mm_loadu_ps(a + i + 4);
		__m128 vb1 = _mm_loadu_ps(b + i + 4);
		__m128 diff1 = _mm_sub_ps(va1, vb1);
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(diff1, diff1));
	}
	__m128 acc = _mm_add_ps(acc0, acc1);

	if (i + 4 <= dim) {
		__m128 va = _mm_loadu_ps(a + i);
		__m128 vb = _mm_loadu_ps(b + i);
		__m128 diff = _mm_sub_ps(va, vb);
		acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
		i += 4;
	}

	/* horizontal add without tmp[] */
	__m128 shuf = _mm_movehdup_ps(acc);        // (3,3,1,1)
	__m128 sums = _mm_add_ps(acc, shuf);
	shuf        = _mm_movehl_ps(shuf, sums);   // (3,1,3,1) → (3,1,*,*)
	sums        = _mm_add_ss(sums, shuf);

	float sum = _mm_cvtss_f32(sums);
#else
	std::size_t i = 0;
	float sum = 0.f;
#endif

#if defined(__AVX512F__)
	for (std::size_t j = i; j < dim; ++j) {
		const float d = a[j] - b[j];
		sum += d * d;
	}
#elif defined(__AVX2__)
	for (std::size_t j = i; j < dim; ++j) {
		const float d = a[j] - b[j];
		sum += d * d;
	}
#elif defined(__SSE2__)
	for (std::size_t j = i; j < dim; ++j) {
		const float d = a[j] - b[j];
		sum += d * d;
	}
#else
	for (std::size_t j = i; j < dim; ++j) {
		const float d = a[j] - b[j];
		sum += d * d;
	}
#endif
	return sum;
}



inline float dot(const float* a, const float* b, std::size_t dim) noexcept {
#if defined(__AVX512F__)
	std::size_t i = 0;
	__m512 acc = _mm512_setzero_ps();
	for (; i + 16 <= dim; i += 16) {
		acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
	}
	alignas(64) float tmp[16];
	_mm512_store_ps(tmp, acc);
	float sum = 0.f;
	for (int j = 0; j < 16; ++j) sum += tmp[j];
#elif defined(__AVX2__)
	std::size_t i = 0;
	__m256 acc = _mm256_setzero_ps();
	for (; i + 8 <= dim; i += 8) {
		acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
	}
	alignas(32) float tmp[8];
	_mm256_store_ps(tmp, acc);
	float sum = 0.f;
	for (int j = 0; j < 8; ++j) sum += tmp[j];
#elif defined(__SSE2__)
	std::size_t i = 0;
	__m128 acc = _mm_setzero_ps();
	for (; i + 4 <= dim; i += 4) {
		acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
	}
	alignas(16) float tmp[4];
	_mm_store_ps(tmp, acc);
	float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#else
	std::size_t i = 0;
	float sum = 0.f;
#endif
	for (std::size_t j = i; j < dim; ++j) sum += a[j] * b[j];
	return sum;
}

inline void l2_sqr_batch(const float* A,
						  const float* B,
						  std::size_t  dim,
						  std::size_t  n,
						  float*       out) noexcept
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
		out[i] = l2_sqr(A + i * dim, B + i * dim, dim);
	}
}

/// out[i] = l2_sqr(Mat[i], v)  (matrix vs. single vector)
inline void l2_sqr_mat_vec(const float* Mat,
							const float* v,
							std::size_t  dim,
							std::size_t  n,
							float*       out) noexcept
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
		out[i] = l2_sqr(Mat + i * dim, v, dim);
	}
}

inline void dot_batch(const float* A,
					   const float* B,
					   std::size_t  dim,
					   std::size_t  n,
					   float*       out) noexcept
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
		out[i] = dot(A + i * dim, B + i * dim, dim);
	}
}

inline void dot_mat_vec(const float* Mat,
						const float* v,
						std::size_t  dim,
						std::size_t  n,
						float*       out) noexcept
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
		out[i] = dot(Mat + i * dim, v, dim);
	}
}


// #########################################

struct LexicalScorerConfig {
	float w_title    = 0.55f;
	float w_attr     = 0.15f;
	float w_subcategory = 0.20f;
	float w_category = 0.10f;

	// Tversky on title. alpha=beta=1 => Jaccard.
	float title_alpha = 1.0f;
	float title_beta  = 1.0f;
	float category_penalty = 1e8f;
};

struct AutocompleteScorerConfig {
	float length_penalty = 0.2f; // the more, the higher
	float position_decay = 0.5f; // position decay, less, more harsh
};


namespace {
	constexpr std::size_t kHeaderSize = 5;
	constexpr std::uint32_t kEncodingVersion = 1;

	constexpr std::size_t kACHeaderSize = 1;
	constexpr std::uint32_t kACEncodingVersion = 1;

	LexicalScorerConfig g_build_cfg{};
	LexicalScorerConfig g_search_cfg{};

	AutocompleteScorerConfig g_ac_build_cfg{};
	AutocompleteScorerConfig g_ac_search_cfg{};

	inline std::size_t clamp_count(std::size_t wanted, std::size_t available) noexcept {
		return wanted < available ? wanted : available;
	}

	inline std::uint32_t as_id(float x) noexcept {
		if (x <= 0.0f) return 0u;
		return static_cast<std::uint32_t>(x);
	}

	inline std::size_t intersection_count_sorted_ids(
		const float* a, std::size_t na,
		const float* b, std::size_t nb
	) noexcept {
		std::size_t i = 0;
		std::size_t j = 0;
		std::size_t inter = 0;

		while (i < na && j < nb) {
			const std::uint32_t va = as_id(a[i]);
			const std::uint32_t vb = as_id(b[j]);

			if (va == vb) {
				if (va != 0u) ++inter;
				++i;
				++j;
			} else if (va < vb) {
				++i;
			} else {
				++j;
			}
		}

		return inter;
	}

	inline float tversky_distance_sorted_ids(
		const float* a, std::size_t na,
		const float* b, std::size_t nb,
		float alpha,
		float beta
	) noexcept {
		if (na == 0 && nb == 0) return 0.0f;

		const std::size_t inter  = intersection_count_sorted_ids(a, na, b, nb);
		const std::size_t only_a = na - inter;
		const std::size_t only_b = nb - inter;

		const float denom =
			static_cast<float>(inter) +
			alpha * static_cast<float>(only_a) +
			beta  * static_cast<float>(only_b);

		if (denom <= 0.0f) return 0.0f;

		const float sim = static_cast<float>(inter) / denom;
		return 1.0f - sim;
	}

	inline float jaccard_distance_sorted_ids(
		const float* a, std::size_t na,
		const float* b, std::size_t nb
	) noexcept {
		return tversky_distance_sorted_ids(a, na, b, nb, 1.0f, 1.0f);
	}


	// prefix matching: how many leading tokens match?
	inline float prefix_distance(
		const float* a_tokens, std::size_t na,
		const float* b_tokens, std::size_t nb,
		const float position_decay,
		const float length_penalty,
		const bool search_time
	) noexcept {
		if (na == 0 || nb == 0) return 1.0f;

		const std::size_t min_len = na < nb ? na : nb;
		// std::size_t matched = 0;
		float pos_score = 0.0f;
		float max_possible = 0.0f;

		for (std::size_t i = 0; i < min_len; ++i) {
			float weight = std::pow(position_decay, (float)i);
			max_possible += weight;
			if (as_id(a_tokens[i]) == as_id(b_tokens[i])) {
				// ++matched;
				pos_score += weight;
			} else {
				break;  // prefix match stops at first mismatch
			}
		}
		float prefix_dist = 1.0f - (pos_score / max_possible);
		if (nb > na and search_time) {  // document longer than query
			float extra_ratio = (float)(nb - na) / (float)na;
			prefix_dist += length_penalty * extra_ratio;
		}
		// const float sim = static_cast<float>(matched) / static_cast<float>(min_len);
		// return (1.0f - sim) + length_penalty + 0.1 * (1.0f - (pos_score / max_possible));
		return prefix_dist;
	}


	inline float id_distance(
		std::uint32_t a_id,
		std::uint32_t b_id,
		float category_penalty
	) noexcept {
		// unknown on either side => do not penalize
		if (a_id == 0u || b_id == 0u) return 0.0f;
		if (a_id == b_id) {
			return 0.0f;
		}
		return category_penalty;
	}

	inline float attribute_distance(
		const float* a_attributes, std::uint32_t a_pairs,
		const float* b_attributes, std::uint32_t b_pairs,
		bool search_time
	) noexcept {
		// no attr on the item side (b), modest penalize
		if (a_pairs != 0u && b_pairs == 0u && search_time) return 0.5f;
		// no attribute on either side => do not penalize
		if (a_pairs == 0u || b_pairs == 0u) return 0.0f;

		std::size_t mismatches = 0;
		std::size_t key_matches = 0;

		if (a_pairs < b_pairs) {
			for (std::size_t i = 0; i < a_pairs; ++i) {
				const std::uint32_t ak = as_id(a_attributes[i * 2]);
				for (std::size_t j = 0; j < b_pairs; ++j) {
					const std::uint32_t bk = as_id(b_attributes[j * 2]);
					 if (ak == bk) {
						++key_matches;
						const std::uint32_t av = as_id(a_attributes[i * 2 + 1]);
						const std::uint32_t bv = as_id(b_attributes[j * 2 + 1]);
						if (av != bv) {
							++mismatches;
						}
						break;
					}
				}
			}
		} else {
			for (std::size_t i = 0; i < b_pairs; ++i) {
				const std::uint32_t bk = as_id(b_attributes[i * 2]);
				for (std::size_t j = 0; j < a_pairs; ++j) {
					const std::uint32_t ak = as_id(a_attributes[j * 2]);
					 if (ak == bk) {
						++key_matches;
						const std::uint32_t av = as_id(a_attributes[j * 2 + 1]);
						const std::uint32_t bv = as_id(b_attributes[i * 2 + 1]);
						if (av != bv) {
							++mismatches;
						}
						break;
					}
				}
			}
		}

		if (mismatches == 0) return 0.0f;

		if (search_time) {
			return 1e8;
			// more penalize during query search
			// return (static_cast<float>(mismatches) / static_cast<float>(a_pairs)) * 2;
		} else {
			if (key_matches == 0) return 0.0f;
			return static_cast<float>(mismatches) / key_matches;
		}
	}

	inline float lexical_structured_distance_impl(
		const float* __restrict a,
		const float* __restrict b,
		std::size_t dim,
		const LexicalScorerConfig& cfg,
		const bool search_time
	) noexcept {
		if (dim < kHeaderSize) return 1.0f;

		const std::size_t payload = dim - kHeaderSize;

		std::size_t a_title_n = clamp_count(static_cast<std::size_t>(a[1]), payload);
		std::size_t b_title_n = clamp_count(static_cast<std::size_t>(b[1]), payload);

		std::size_t a_attr_n = clamp_count(static_cast<std::size_t>(a[2]), payload - a_title_n);
		std::size_t b_attr_n = clamp_count(static_cast<std::size_t>(b[2]), payload - b_title_n);

		const std::uint32_t a_category = as_id(a[3]);
		const std::uint32_t b_category = as_id(b[3]);

		const std::uint32_t a_subcategory    = as_id(a[4]);
		const std::uint32_t b_subcategory    = as_id(b[4]);


		const float* a_title = a + kHeaderSize;
		const float* b_title = b + kHeaderSize;

		const float* a_attr = a_title + a_title_n;
		const float* b_attr = b_title + b_title_n;

		const float title_dist = tversky_distance_sorted_ids(
			a_title, a_title_n,
			b_title, b_title_n,
			cfg.title_alpha, cfg.title_beta
		);

		// keep attrs simple for now
		const float attr_dist = attribute_distance(a_attr, a_attr_n, b_attr, b_attr_n, search_time);

		const float category_dist   = id_distance(a_category, b_category, cfg.category_penalty);
		const float subcategory_dist = id_distance(a_subcategory, b_subcategory, cfg.category_penalty);

		return
			cfg.w_title    * title_dist +
			cfg.w_attr     * attr_dist +
			cfg.w_subcategory * subcategory_dist +
			cfg.w_category * category_dist;
	}

	inline float autocomplete_distance_impl(
		const float* __restrict a,
		const float* __restrict b,
		std::size_t dim,
		const AutocompleteScorerConfig& cfg,
		const bool search_time
	) noexcept {
		if (dim < kACHeaderSize) return 1.0f;

		const std::size_t payload = dim - kACHeaderSize;

		// header: [token_count, ...tokens]
		std::size_t a_token_n = clamp_count(static_cast<std::size_t>(a[0]), payload);
		std::size_t b_token_n = clamp_count(static_cast<std::size_t>(b[0]), payload);

		const float* a_tokens = a + kACHeaderSize;
		const float* b_tokens = b + kACHeaderSize;

		const float prefix_dist = prefix_distance(
			a_tokens, a_token_n, b_tokens, b_token_n,
			cfg.position_decay, cfg.length_penalty,
			search_time
		);
		return prefix_dist;

	}

} // namespace

void set_build_lexical_config(const LexicalScorerConfig& cfg) {
	g_build_cfg = cfg;
}

void set_search_lexical_config(const LexicalScorerConfig& cfg) {
	g_search_cfg = cfg;
}

void set_build_autocomplete_config(const AutocompleteScorerConfig& cfg) {
	g_ac_build_cfg = cfg;
}

void set_search_autocomplete_config(const AutocompleteScorerConfig& cfg) {
	g_ac_search_cfg = cfg;
}

float lexical_structured_distance_build(
	const float* __restrict a,
	const float* __restrict b,
	std::size_t dim
) noexcept {
	return lexical_structured_distance_impl(a, b, dim, g_build_cfg, false);
}

float lexical_structured_distance_search(
	const float* __restrict a,
	const float* __restrict b,
	std::size_t dim
) noexcept {
	return lexical_structured_distance_impl(a, b, dim, g_search_cfg, true);
}

float autocomplete_distance_build(
	const float* __restrict a,
	const float* __restrict b,
	std::size_t dim
) noexcept {
	return autocomplete_distance_impl(a, b, dim, g_ac_build_cfg, false);
}

float autocomplete_distance_search(
	const float* __restrict a,
	const float* __restrict b,
	std::size_t dim
) noexcept {
	return autocomplete_distance_impl(a, b, dim, g_ac_search_cfg, true);
}



}
