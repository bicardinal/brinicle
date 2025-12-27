import os
import shutil
import tempfile
import time
import numpy as np
from threading import Thread, Barrier, Event
from multiprocessing import Process, Queue
import multiprocessing as mp

class VectorEngineConcurrentSearchTests:
	def __init__(self, VectorEngine):
		self.VectorEngine = VectorEngine
		self.test_dir = None

	def setup(self):
		self.test_dir = tempfile.mkdtemp(prefix="VectorEngine_search_test_")
		print(f"Test directory: {self.test_dir}")

	def teardown(self):
		if self.test_dir and os.path.exists(self.test_dir):
			shutil.rmtree(self.test_dir)
			print(f"Cleaned up: {self.test_dir}")

	def _get_test_path(self, name):
		return os.path.join(self.test_dir, name)

	def _generate_random_vectors(self, n, dim):
		vecs = np.random.randn(n, dim).astype(np.float32)
		norms = np.linalg.norm(vecs, axis=1, keepdims=True)
		return vecs / (norms + 1e-8)

	def _build_initial_index(self, index_path, dim, n_vectors=1000):
		db = self.VectorEngine(index_path, dim)
		db.init("build")
		vectors = self._generate_random_vectors(n_vectors, dim)
		for i, vec in enumerate(vectors):
			db.ingest(f"vec_{i}", vec)
		db.finalize()
		db.close()
		return vectors

	def test_search_heavy_load(self):
		print("\n=== Heavy Search Load ===")

		index_path = self._get_test_path("search_heavy_load")
		dim = 128
		n_vectors = 5000

		self._build_initial_index(index_path, dim, n_vectors)

		search_counts = []
		errors = []
		start_barrier = Barrier(20)

		def search_worker(thread_id, n_searches, k):
			try:
				start_barrier.wait()
				local_db = self.VectorEngine(index_path, dim)
				count = 0

				start_time = time.time()
				for i in range(n_searches):
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=k)
					assert len(res) > 0, f"Thread {thread_id}: Empty results at iteration {i}"
					assert len(res) <= k, f"Thread {thread_id}: Too many results ({len(res)} > {k})"
					count += 1

				elapsed = time.time() - start_time
				local_db.close()
				search_counts.append((thread_id, count, elapsed))
			except Exception as e:
				errors.append((thread_id, str(e)))

		n_threads = 20
		searches_per_thread = 200
		k = 50
		threads = []

		print(f"Launching {n_threads} threads, {searches_per_thread} searches each...")

		for i in range(n_threads):
			t = Thread(target=search_worker, args=(i, searches_per_thread, k))
			threads.append(t)
			t.start()

		for t in threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(search_counts) == n_threads, f"Not all threads completed"

		total_searches = sum(c[1] for c in search_counts)
		avg_time = sum(c[2] for c in search_counts) / len(search_counts)

		print(f"[OK] {total_searches} searches completed successfully")
		print(f"  Average time per thread: {avg_time:.2f}s")

	def test_search_during_insert(self):
		print("\n=== Search During Insert ===")

		index_path = self._get_test_path("search_insert")
		dim = 96

		self._build_initial_index(index_path, dim, 500)

		errors = []
		search_results = []
		insert_results = []
		stop_event = Event()

		def continuous_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0
				empty_count = 0

				while not stop_event.is_set():
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=10)
					if len(res) == 0:
						empty_count += 1
					count += 1
					time.sleep(0.001)

				local_db.close()
				search_results.append((thread_id, count, empty_count))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def batch_inserter(batch_id, n_batches, vectors_per_batch):
			try:
				time.sleep(0.1)

				for batch_num in range(n_batches):
					local_db = self.VectorEngine(index_path, dim)
					local_db.init("insert")

					for i in range(vectors_per_batch):
						vec = self._generate_random_vectors(1, dim)[0]
						vec_id = f"batch_{batch_id}_round_{batch_num}_vec_{i}"
						local_db.ingest(vec_id, vec)

					local_db.finalize()
					local_db.close()
					time.sleep(0.05)

				insert_results.append((batch_id, n_batches * vectors_per_batch))
			except Exception as e:
				errors.append(("inserter", batch_id, str(e)))

		n_searchers = 8
		searcher_threads = []
		for i in range(n_searchers):
			t = Thread(target=continuous_searcher, args=(i,))
			t.start()
			searcher_threads.append(t)

		n_inserters = 100
		batches_per_inserter = 5
		vectors_per_batch = 20
		inserter_threads = []

		for i in range(n_inserters):
			t = Thread(target=batch_inserter, args=(i, batches_per_inserter, vectors_per_batch))
			t.start()
			inserter_threads.append(t)

		for t in inserter_threads:
			t.join()

		time.sleep(0.2)
		stop_event.set()

		for t in searcher_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(insert_results) == n_inserters, "Not all inserters completed"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)
		total_inserts = sum(r[1] for r in insert_results)
		total_empty = sum(r[2] for r in search_results)

		print(f"[OK] {total_searches} searches during {total_inserts} inserts")
		print(f"  Empty results: {total_empty} ({100*total_empty/total_searches:.2f}%)")

	def test_search_during_delete(self):
		print("\n=== Search During Delete ===")

		index_path = self._get_test_path("search_delete")
		dim = 80
		n_vectors = 2000

		self._build_initial_index(index_path, dim, n_vectors)

		errors = []
		search_results = []
		delete_results = []
		stop_event = Event()

		def continuous_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0
				result_sizes = []

				while not stop_event.is_set():
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=20)
					result_sizes.append(len(res))
					count += 1
					time.sleep(0.002)

				local_db.close()
				avg_results = sum(result_sizes) / len(result_sizes) if result_sizes else 0
				search_results.append((thread_id, count, avg_results))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def batch_deleter(deleter_id, n_rounds, delete_per_round):
			try:
				time.sleep(0.15)

				for round_num in range(n_rounds):
					local_db = self.VectorEngine(index_path, dim)

					# Delete a batch of IDs
					start_id = deleter_id * 400 + round_num * delete_per_round
					ids_to_delete = [f"vec_{i}" for i in range(start_id, start_id + delete_per_round)]

					deleted_count, not_found = local_db.delete_items(ids_to_delete, return_not_found=True)

					local_db.close()
					time.sleep(0.08)

				delete_results.append((deleter_id, n_rounds * delete_per_round))
			except Exception as e:
				errors.append(("deleter", deleter_id, str(e)))

		n_searchers = 10
		searcher_threads = []
		for i in range(n_searchers):
			t = Thread(target=continuous_searcher, args=(i,))
			t.start()
			searcher_threads.append(t)

		n_deleters = 3
		rounds_per_deleter = 6
		delete_per_round = 50
		deleter_threads = []

		for i in range(n_deleters):
			t = Thread(target=batch_deleter, args=(i, rounds_per_deleter, delete_per_round))
			t.start()
			deleter_threads.append(t)

		for t in deleter_threads:
			t.join()

		time.sleep(0.2)
		stop_event.set()

		for t in searcher_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(delete_results) == n_deleters, "Not all deleters completed"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)
		total_deletions = sum(r[1] for r in delete_results)
		avg_result_size = sum(r[2] for r in search_results) / len(search_results)

		print(f"[OK] {total_searches} searches during {total_deletions} deletions")
		print(f"  Average search results: {avg_result_size:.1f}")

	def test_search_during_upsert(self):
		print("\n=== Search During Upsert ===")

		index_path = self._get_test_path("search_upsert")
		dim = 64
		n_vectors = 800

		self._build_initial_index(index_path, dim, n_vectors)

		errors = []
		search_results = []
		upsert_results = []
		stop_event = Event()

		def continuous_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0

				while not stop_event.is_set():
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=15)
					count += 1
					time.sleep(0.003)

				local_db.close()
				search_results.append((thread_id, count))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def batch_upserter(upserter_id, n_rounds, upsert_per_round):
			try:
				time.sleep(0.1)

				for round_num in range(n_rounds):
					local_db = self.VectorEngine(index_path, dim)
					local_db.init("upsert")

					start_id = upserter_id * 200 + (round_num * 30)
					for i in range(upsert_per_round):
						vec = self._generate_random_vectors(1, dim)[0]
						vec_id = f"vec_{(start_id + i) % n_vectors}"
						local_db.ingest(vec_id, vec)

					local_db.finalize()
					local_db.close()
					time.sleep(0.1)

				upsert_results.append((upserter_id, n_rounds * upsert_per_round))
			except Exception as e:
				errors.append(("upserter", upserter_id, str(e)))

		n_searchers = 6
		searcher_threads = []
		for i in range(n_searchers):
			t = Thread(target=continuous_searcher, args=(i,))
			t.start()
			searcher_threads.append(t)

		n_upserters = 3
		rounds_per_upserter = 4
		upsert_per_round = 40
		upserter_threads = []

		for i in range(n_upserters):
			t = Thread(target=batch_upserter, args=(i, rounds_per_upserter, upsert_per_round))
			t.start()
			upserter_threads.append(t)

		for t in upserter_threads:
			t.join()

		time.sleep(0.2)
		stop_event.set()

		for t in searcher_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(upsert_results) == n_upserters, "Not all upserters completed"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)
		total_upserts = sum(r[1] for r in upsert_results)

		print(f"[OK] {total_searches} searches during {total_upserts} upserts")

	def test_search_during_build(self):
		print("\n=== Search During Build ===")

		index_path = self._get_test_path("search_build")
		dim = 96
		n_initial = 600

		self._build_initial_index(index_path, dim, n_initial)

		errors = []
		search_results = []
		build_complete = []
		stop_event = Event()

		def continuous_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0
				result_sizes = []

				while not stop_event.is_set():
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=10)
					result_sizes.append(len(res))
					count += 1
					time.sleep(0.005)

				local_db.close()
				avg_size = sum(result_sizes) / len(result_sizes) if result_sizes else 0
				search_results.append((thread_id, count, avg_size))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def index_builder(n_new_vectors):
			try:
				time.sleep(0.1)

				local_db = self.VectorEngine(index_path, dim)
				local_db.init("build")

				for i in range(n_new_vectors):
					vec = self._generate_random_vectors(1, dim)[0]
					local_db.ingest(f"newvec_{i}", vec)
					if i % 100 == 0:
						time.sleep(0.01)

				local_db.finalize()
				local_db.close()
				build_complete.append(n_new_vectors)
			except Exception as e:
				errors.append(("builder", str(e)))

		n_searchers = 8
		searcher_threads = []
		for i in range(n_searchers):
			t = Thread(target=continuous_searcher, args=(i,))
			t.start()
			searcher_threads.append(t)

		n_new_vectors = 1200
		builder_thread = Thread(target=index_builder, args=(n_new_vectors,))
		builder_thread.start()

		builder_thread.join()

		time.sleep(0.3)
		stop_event.set()

		for t in searcher_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(build_complete) == 1, "Builder did not complete"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)

		print(f"[OK] {total_searches} searches during full rebuild of {n_new_vectors} vectors")
		print(f"  Build completed successfully")

	def test_search_during_rebuild_compact(self):
		print("\n=== Search During Rebuild Compact ===")

		index_path = self._get_test_path("search_rebuild_compact")
		dim = 72

		self._build_initial_index(index_path, dim, 1000)

		db = self.VectorEngine(index_path, dim)
		db.init("insert")
		for i in range(200):
			vec = self._generate_random_vectors(1, dim)[0]
			db.ingest(f"extra_{i}", vec)
		db.finalize()

		ids_to_delete = [f"vec_{i}" for i in range(0, 500, 5)]
		db.delete_items(ids_to_delete)
		db.close()

		errors = []
		search_results = []
		rebuild_complete = []
		stop_event = Event()

		def continuous_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0
				empty_results = 0

				while not stop_event.is_set():
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=10)
					if len(res) == 0:
						empty_results += 1
					count += 1
					time.sleep(0.003)

				local_db.close()
				search_results.append((thread_id, count, empty_results))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def index_rebuilder():
			try:
				time.sleep(0.15)

				local_db = self.VectorEngine(index_path, dim)
				local_db.rebuild_compact()
				local_db.close()
				rebuild_complete.append(True)
			except Exception as e:
				errors.append(("rebuilder", str(e)))

		n_searchers = 6
		searcher_threads = []
		for i in range(n_searchers):
			t = Thread(target=continuous_searcher, args=(i,))
			t.start()
			searcher_threads.append(t)

		rebuilder_thread = Thread(target=index_rebuilder)
		rebuilder_thread.start()

		rebuilder_thread.join()

		time.sleep(0.3)
		stop_event.set()

		for t in searcher_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(rebuild_complete) == 1, "Rebuild did not complete"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)
		total_empty = sum(r[2] for r in search_results)

		print(f"[OK] {total_searches} searches during rebuild_compact")
		print(f"  Empty results: {total_empty}")

	def test_search_during_optimize(self):
		print("\n=== Search During Optimize ===")

		index_path = self._get_test_path("search_optimize")
		dim = 64

		self._build_initial_index(index_path, dim, 500)

		db = self.VectorEngine(index_path, dim, delta_ratio=0.2)
		db.init("insert")
		for i in range(150):
			vec = self._generate_random_vectors(1, dim)[0]
			db.ingest(f"delta_{i}", vec)
		db.finalize(optimize=True)
		db.close()

		errors = []
		search_results = []
		optimize_complete = []
		stop_event = Event()

		def continuous_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim, delta_ratio=0.2)
				count = 0

				while not stop_event.is_set():
					query = self._generate_random_vectors(1, dim)[0]
					res = local_db.search(query, k=10)
					count += 1
					time.sleep(0.004)

				local_db.close()
				search_results.append((thread_id, count))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def optimizer():
			try:
				time.sleep(0.1)

				local_db = self.VectorEngine(index_path, dim, delta_ratio=0.2)

				needs = local_db.needs_rebuild()
				if needs:
					local_db.optimize_graph()

				local_db.close()
				optimize_complete.append(needs)
			except Exception as e:
				errors.append(("optimizer", str(e)))

		n_searchers = 5
		searcher_threads = []
		for i in range(n_searchers):
			t = Thread(target=continuous_searcher, args=(i,))
			t.start()
			searcher_threads.append(t)

		optimizer_thread = Thread(target=optimizer)
		optimizer_thread.start()

		optimizer_thread.join()

		time.sleep(0.3)
		stop_event.set()

		for t in searcher_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)

		print(f"[OK] {total_searches} searches during optimize_graph")
		print(f"  Optimization triggered: {optimize_complete[0] if optimize_complete else 'N/A'}")

	def test_search_multi_operation_stress(self):
		print("\n=== Search Multi-Operation Stress ===")

		index_path = self._get_test_path("search_stress")
		dim = 80

		self._build_initial_index(index_path, dim, 1500)

		errors = []
		search_results = []
		operation_results = {"insert": [], "delete": [], "upsert": []}
		stop_event = Event()

		def intensive_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0
				errors_local = 0

				while not stop_event.is_set():
					try:
						query = self._generate_random_vectors(1, dim)[0]
						res = local_db.search(query, k=20)
						count += 1
					except Exception as e:
						errors_local += 1

					time.sleep(0.001)

				local_db.close()
				search_results.append((thread_id, count, errors_local))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def inserter_worker(worker_id, n_ops):
			try:
				time.sleep(0.1)
				for op in range(n_ops):
					local_db = self.VectorEngine(index_path, dim)
					local_db.init("insert")
					for i in range(10):
						vec = self._generate_random_vectors(1, dim)[0]
						local_db.ingest(f"insert_w{worker_id}_op{op}_v{i}", vec)
					local_db.finalize()
					local_db.close()
					time.sleep(0.05)
				operation_results["insert"].append(worker_id)
			except Exception as e:
				errors.append(("inserter", worker_id, str(e)))

		def deleter_worker(worker_id, n_ops):
			try:
				time.sleep(0.15)
				for op in range(n_ops):
					local_db = self.VectorEngine(index_path, dim)
					start_id = worker_id * 200 + op * 20
					ids = [f"vec_{i}" for i in range(start_id, start_id + 20)]
					local_db.delete_items(ids)
					local_db.close()
					time.sleep(0.06)
				operation_results["delete"].append(worker_id)
			except Exception as e:
				errors.append(("deleter", worker_id, str(e)))

		def upserter_worker(worker_id, n_ops):
			try:
				time.sleep(0.2)
				for op in range(n_ops):
					local_db = self.VectorEngine(index_path, dim)
					local_db.init("upsert")
					start_id = worker_id * 100
					for i in range(15):
						vec = self._generate_random_vectors(1, dim)[0]
						local_db.ingest(f"vec_{start_id + i}", vec)
					local_db.finalize()
					local_db.close()
					time.sleep(0.08)
				operation_results["upsert"].append(worker_id)
			except Exception as e:
				errors.append(("upserter", worker_id, str(e)))

		threads = []

		n_searchers = 15
		for i in range(n_searchers):
			t = Thread(target=intensive_searcher, args=(i,))
			t.start()
			threads.append(t)

		for i in range(3):
			t = Thread(target=inserter_worker, args=(i, 4))
			t.start()
			threads.append(t)

		for i in range(2):
			t = Thread(target=deleter_worker, args=(i, 3))
			t.start()
			threads.append(t)

		for i in range(2):
			t = Thread(target=upserter_worker, args=(i, 3))
			t.start()
			threads.append(t)

		time.sleep(2.5)
		stop_event.set()

		for t in threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(search_results) == n_searchers, "Not all searchers completed"

		total_searches = sum(r[1] for r in search_results)
		total_search_errors = sum(r[2] for r in search_results)

		print(f"[OK] {total_searches} searches during mixed operations")
		print(f"  Inserters completed: {len(operation_results['insert'])}")
		print(f"Deleters completed: {len(operation_results['delete'])}")
		print(f"  Upserters completed: {len(operation_results['upsert'])}")
		print(f"  Search errors: {total_search_errors}")

	def test_search_correctness_during_modifications(self):
		print("\n=== Search Correctness During Modifications ===")

		index_path = self._get_test_path("search_correctness")
		dim = 32

		db = self.VectorEngine(index_path, dim)
		db.init("build")

		base_vectors = self._generate_random_vectors(500, dim)
		for i, vec in enumerate(base_vectors):
			db.ingest(f"base_{i}", vec)

		db.finalize()
		db.close()

		errors = []
		validity_checks = []
		stop_event = Event()

		def validating_searcher(thread_id):
			try:
				local_db = self.VectorEngine(index_path, dim)
				count = 0
				invalid = 0

				while not stop_event.is_set():
					query_idx = count % len(base_vectors)
					query = base_vectors[query_idx]

					res = local_db.search(query, k=5)

					if len(res) == 0:
						invalid += 1

					if len(res) != len(set(res)):
						invalid += 1

					count += 1
					time.sleep(0.005)

				local_db.close()
				validity_checks.append((thread_id, count, invalid))
			except Exception as e:
				errors.append(("searcher", thread_id, str(e)))

		def modifier_worker():
			try:
				time.sleep(0.1)

				for round_num in range(4):
					local_db = self.VectorEngine(index_path, dim)

					local_db.init("insert")
					for i in range(20):
						vec = self._generate_random_vectors(1, dim)[0]
						local_db.ingest(f"new_r{round_num}_v{i}", vec)
					local_db.finalize()

					ids = [f"base_{i}" for i in range(round_num * 10, round_num * 10 + 5)]
					local_db.delete_items(ids)

					local_db.close()
					time.sleep(0.15)

			except Exception as e:
				errors.append(("modifier", str(e)))

		n_validators = 6
		validator_threads = []
		for i in range(n_validators):
			t = Thread(target=validating_searcher, args=(i,))
			t.start()
			validator_threads.append(t)

		modifier_thread = Thread(target=modifier_worker)
		modifier_thread.start()

		modifier_thread.join()

		time.sleep(0.2)
		stop_event.set()

		for t in validator_threads:
			t.join()

		assert len(errors) == 0, f"Errors occurred: {errors}"
		assert len(validity_checks) == n_validators, "Not all validators completed"

		total_checks = sum(v[1] for v in validity_checks)
		total_invalid = sum(v[2] for v in validity_checks)
		invalid_rate = 100.0 * total_invalid / total_checks if total_checks > 0 else 0

		print(f"[OK] {total_checks} validity checks performed")
		print(f"  Invalid results: {total_invalid} ({invalid_rate:.2f}%)")

		assert invalid_rate < 10.0, f"Too many invalid results: {invalid_rate:.2f}%"

	def run_all_tests(self):
		print("\n" + "="*70)
		print("VectorEngine Concurrent Search Test Suite")
		print("="*70)

		self.setup()

		try:
			self.test_search_heavy_load()
			self.test_search_during_insert()
			self.test_search_during_delete()
			self.test_search_during_upsert()
			self.test_search_during_build()
			self.test_search_during_rebuild_compact()
			self.test_search_during_optimize()
			self.test_search_multi_operation_stress()
			self.test_search_correctness_during_modifications()

			print("\n" + "="*70)
			print("[OK] All concurrent search tests passed!")
			print("="*70)

		finally:
			self.teardown()


if __name__ == "__main__":
	from brinicle import VectorEngine

	test_suite = VectorEngineConcurrentSearchTests(VectorEngine)
	test_suite.run_all_tests()
