
#pragma once
#ifndef L2_HSUM_UTILS_H
#define L2_HSUM_UTILS_H

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

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

}
