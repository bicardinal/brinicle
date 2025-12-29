#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hnsw.h"
#include "hnsw_manager.h"

#include "knn.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;
using ops::BruteForceKNN;


static inline void check_f32_2d(const py::array &a, const char* name) {
	if (!py::isinstance<py::array>(a)) throw std::runtime_error(std::string(name) + " must be ndarray");
	if (a.ndim() != 2) throw std::runtime_error(std::string(name) + " must be 2-D float32");
}

static inline void check_f32_1d(const py::array &a, const char* name) {
	if (!py::isinstance<py::array>(a)) throw std::runtime_error(std::string(name) + " must be ndarray");
	if (a.ndim() != 1) throw std::runtime_error(std::string(name) + " must be 1-D float32");
}

PYBIND11_MODULE(_brinicle, m) {
	m.doc() = "HNSW ANN bindings";


	m.def("l2_sqr", 
	[](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
		const py::array_t<float, py::array::c_style | py::array::forcecast>& b) -> float {
		if (a.ndim() != 1 || b.ndim() != 1) {
			throw std::runtime_error("a and b must be 1-D float32 arrays");
		}
		if (a.shape(0) != b.shape(0)) {
			throw std::runtime_error("a and b must have the same length");
		}
		const float* a_ptr = static_cast<const float*>(a.request().ptr);
		const float* b_ptr = static_cast<const float*>(b.request().ptr);
		std::size_t dim = static_cast<std::size_t>(a.shape(0));
		return ops::l2_sqr(a_ptr, b_ptr, dim);
	},
	py::arg("a"), py::arg("b"),
	"Compute squared L2 distance between two vectors"
	);


	m.def("dot",
	[](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
	   const py::array_t<float, py::array::c_style | py::array::forcecast>& b) -> float {
		if (a.ndim() != 1 || b.ndim() != 1) {
			throw std::runtime_error("a and b must be 1-D float32 arrays");
		}
		if (a.shape(0) != b.shape(0)) {
			throw std::runtime_error("a and b must have the same length");
		}

		const float* a_ptr = static_cast<const float*>(a.request().ptr);
		const float* b_ptr = static_cast<const float*>(b.request().ptr);
		std::size_t dim = static_cast<std::size_t>(a.shape(0));
		return ops::dot(a_ptr, b_ptr, dim);
	},
	py::arg("a"), py::arg("b"),
	"Compute dot product between two vectors"
	);


	m.def("brute_knn_batch",
	[](const py::array_t<float, py::array::c_style | py::array::forcecast>& X,
	   const py::array_t<float, py::array::c_style | py::array::forcecast>& Q,
	   std::size_t k,
	   int n_jobs)
	{
		if (X.ndim() != 2 || Q.ndim() != 2) throw std::runtime_error("X and Q must be 2-D float32");
		if (X.shape(1) != Q.shape(1)) throw std::runtime_error("Dim mismatch between X and Q");
		if (k == 0) throw std::runtime_error("k must be ≥ 1");

		const std::size_t N   = X.shape(0);
		const std::size_t dim = X.shape(1);
		const std::size_t Qn  = Q.shape(0);

		k = std::min<std::size_t>(k, N);

		auto out_idx  = py::array_t<int32_t>  ({Qn, k});
		auto out_dist = py::array_t<float>    ({Qn, k});

		const float* Xptr = static_cast<const float*>(X.request().ptr);
		const float* Qptr = static_cast<const float*>(Q.request().ptr);
		int32_t* idx_ptr  = static_cast<int32_t*>(out_idx.request().ptr);
		float*   dst_ptr  = static_cast<float*>(out_dist.request().ptr);

		ops::BruteForceKNN bf(Xptr, N, dim);

#ifdef _OPENMP
	  const int prev = omp_get_max_threads();
	  if (n_jobs > 0) omp_set_num_threads(n_jobs);
#endif
	  {
		py::gil_scoped_release rel;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (std::ptrdiff_t qi = 0; qi < static_cast<std::ptrdiff_t>(Qn); ++qi) {
			std::vector<int>   ids;
			std::vector<float> dists;
			bf.query(Qptr + qi * dim, k, ids, dists);
			std::memcpy(idx_ptr + qi * k, ids.data(),   k * sizeof(int32_t));
			std::memcpy(dst_ptr + qi * k, dists.data(), k * sizeof(float));
		}
	  }
#ifdef _OPENMP
	  if (n_jobs > 0) omp_set_num_threads(prev);
#endif
	  return py::make_tuple(out_idx, out_dist);
	},
	py::arg("X"), py::arg("Q"), py::arg("k"), py::arg("n_jobs") = -1
	);


	py::class_<ghnsw::Params>(m, "HNSWParams")
		.def(py::init<>())
		.def_readwrite("M", &ghnsw::Params::M)
		.def_readwrite("ef_construction", &ghnsw::Params::ef_construction)
		.def_readwrite("ef_search", &ghnsw::Params::ef_search)
		.def_readwrite("rng_seed", &ghnsw::Params::rng_seed);


	py::class_<ghnsw_mgr::VectorEngine>(m, "VectorEngine")
		.def(py::init([](const std::string& index_path,
					 std::size_t dim,
					 const float delta_ratio,
					 std::size_t M,
					 std::size_t ef_construction,
					 std::size_t ef_search,
					 std::size_t seed) {
			ghnsw::Params params;
			params.M = M;
			params.ef_construction = ef_construction;
			params.ef_search = ef_search;
			params.rng_seed = seed;
			py::gil_scoped_release rel;
			return std::make_unique<ghnsw_mgr::VectorEngine>(index_path, dim, delta_ratio, params);
		}),
		py::arg("index_path"),
		py::arg("dim") = 0,
		py::arg("delta_ratio") = 0.10,
		py::arg("M") = 16,
		py::arg("ef_construction") = 200,
		py::arg("ef_search") = 64,
		py::arg("seed") = 0)


		.def("init", [](ghnsw_mgr::VectorEngine& self,
						const std::string& mode) {
			py::gil_scoped_release rel;
			self.init(mode);
		},
		py::arg("mode"),
		"Start incremental ingest. mode in ['build', 'insert', 'upsert']")


		.def("ingest", [](ghnsw_mgr::VectorEngine& self,
						  const std::string& external_id,
						  const py::array& vec) {
			check_f32_1d(vec, "vec");
			auto a = py::array_t<float, py::array::c_style | py::array::forcecast>(vec);
			if ((std::size_t)a.shape(0) != self.dim())
			throw std::runtime_error("ingest: dimension mismatch");
			py::gil_scoped_release rel;
			self.ingest(external_id, a.data());
		},
		py::arg("external_id"),
		py::arg("vec"),
		"Append one vector to the current ingest file")


		.def("finalize", [](ghnsw_mgr::VectorEngine& self,
						const bool optimize,
						std::size_t M,
						std::size_t ef_construction,
						std::size_t ef_search,
						std::size_t seed) {
			ghnsw::Params params;
			params.M = M;
			params.ef_construction = ef_construction;
			params.ef_search = ef_search;
			params.rng_seed = seed;
			py::gil_scoped_release rel;
			self.finalize(params, optimize);
		},
		py::arg("optimize") = false,
		py::arg("M") = 0, // it means use the initialized params as default
		py::arg("ef_construction") = 0,
		py::arg("ef_search") = 0,
		py::arg("seed") = 0,
		"Finalize ingest and build/insert/upsert depending on init(mode).")


		.def("build_from_file", [](ghnsw_mgr::VectorEngine& self,
								   const std::string& vectors_path,
								   const std::vector<std::string>& external_ids,
								   const ghnsw::Params& build_params) {
			py::gil_scoped_release rel;
			self.build_from_file(vectors_path, external_ids, build_params);
		},
		py::arg("vectors_path"),
		py::arg("external_ids"),
		py::arg("build_params") = ghnsw::Params(),
		"Build from a vector file path.")


		.def("delete_items", [](ghnsw_mgr::VectorEngine& self,
								const std::vector<std::string>& external_ids,
								bool return_not_found) {
			std::vector<std::string> not_found;
			std::size_t n = 0;
			{
				py::gil_scoped_release rel;
				n = self.delete_external_ids(external_ids, return_not_found ? &not_found : nullptr);
			}
			if (return_not_found) return py::make_tuple(n, not_found);
			return py::make_tuple(n, py::none());
		},
		py::arg("external_ids"),
		py::arg("return_not_found") = false,
		"Delete items by id.")


		.def("rebuild_compact", [](ghnsw_mgr::VectorEngine& self,
						std::size_t M = 16,
						std::size_t ef_construction = 200,
						std::size_t ef_search = 64,
						std::size_t seed = 0) {
			ghnsw::Params params;
			params.M = M;
			params.ef_construction = ef_construction;
			params.ef_search = ef_search;
			params.rng_seed = seed;
			py::gil_scoped_release rel;
			self.rebuild_compact(params);
		},
		py::arg("M") = 16,
		py::arg("ef_construction") = 16,
		py::arg("ef_search") = 64,
		py::arg("seed") = 0,
		"Rebuild the index and clean up segments.")


		.def("search", [](ghnsw_mgr::VectorEngine& self,
						  const py::array& q,
						  int k,
						  int efs) {
			check_f32_1d(q, "q");
			auto a = py::array_t<float, py::array::c_style | py::array::forcecast>(q);
			if ((std::size_t)a.shape(0) != self.dim())
				throw std::runtime_error("search: dimension mismatch");

			std::vector<std::string> out;
			{
				py::gil_scoped_release rel;
				out = self.search(a.data(), k, efs);
			}
			return out;
		},
		py::arg("q"),
		py::arg("k") = 10,
		py::arg("efs") = 64,
		"Search, and return external IDs")


		.def("needs_rebuild", &ghnsw_mgr::VectorEngine::needs_rebuild)
		.def("optimize_graph", &ghnsw_mgr::VectorEngine::optimize_graph)
		.def("close", &ghnsw_mgr::VectorEngine::close)
		.def("destroy", &ghnsw_mgr::VectorEngine::destroy)
		.def_property_readonly("dim", &ghnsw_mgr::VectorEngine::dim)
		.def_property_readonly("has_index", &ghnsw_mgr::VectorEngine::has_index);
}
