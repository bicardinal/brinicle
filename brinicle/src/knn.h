#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "operations.h"

namespace ops {

class BruteForceKNN {
public:
	BruteForceKNN(const float* data, std::size_t n, std::size_t dim) noexcept
		: data_(data), n_(n), dim_(dim) {}

	// exact k‑NN of external query vector q.
	void query(const float* q,
			   std::size_t k,
			   std::vector<int>&   out_idx,
			   std::vector<float>& out_dist) const
	{
		if (n_ == 0 || k == 0) {
			out_idx.clear();
			out_dist.clear();
			return;
		}
		k = std::min(k, n_);

		using Pair = std::pair<float, int>;
		std::priority_queue<Pair, std::vector<Pair>, std::less<>> heap;
		heap = std::priority_queue<Pair, std::vector<Pair>, std::less<>>();

		for (std::size_t j = 0; j < n_; ++j) {
			const float d = l2_sqr(q, data_ + j * dim_, dim_);
			if (heap.size() < k) {
				heap.emplace(d, static_cast<int>(j));
			} else if (d < heap.top().first) {
				heap.pop();
				heap.emplace(d, static_cast<int>(j));
			}
		}

		out_idx.resize(k);
		out_dist.resize(k);
		for (std::size_t t = k; t-- > 0; ) {
			const auto& p = heap.top();
			out_idx[t]  = p.second;
			out_dist[t] = p.first;
			heap.pop();
		}
	}

	void all_knn(std::size_t k, std::vector<std::vector<int>>& indices) const {
		if (n_ == 0 || k == 0) {
			indices.clear();
			return;
		}
		k = std::min(k, n_ - 1); // exclude self
		indices.assign(n_, std::vector<int>(k));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_); ++i) {
			std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>, std::less<>> heap;
			const float* xi = data_ + i * dim_;
			for (std::size_t j = 0; j < n_; ++j) {
				if (j == static_cast<std::size_t>(i)) continue; // skip self
				const float d = l2_sqr(xi, data_ + j * dim_, dim_);
				if (heap.size() < k) {
					heap.emplace(d, static_cast<int>(j));
				} else if (d < heap.top().first) {
					heap.pop();
					heap.emplace(d, static_cast<int>(j));
				}
			}
			for (std::size_t t = k; t-- > 0; ) {
				indices[i][t] = heap.top().second;
				heap.pop();
			}
		}
	}

private:
	const float* data_;      // pointer to row‑major (N×dim) float array (external lifecycle)
	std::size_t  n_;         // number of vectors
	std::size_t  dim_;       // dimensionality
};

}