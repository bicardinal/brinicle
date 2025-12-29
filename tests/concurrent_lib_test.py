import os
import shutil
import tempfile
import time
import traceback
from multiprocessing import Queue
from threading import Barrier
from threading import Thread

import numpy as np


class VectorEngineConcurrentTests:

    def __init__(self, VectorEngine):
        self.VectorEngine = VectorEngine
        self.test_dir = None

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="VectorEngine_test_")
        print(f"Test directory: {self.test_dir}")

    def teardown(self):
        if self.test_dir and os.path.exists(self.test_dir):
            time.sleep(0.2)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(self.test_dir)
                    print(f"Cleaned up: {self.test_dir}")
                    break
                except (OSError, PermissionError) as e:
                    if attempt < max_retries - 1:
                        print(f"Cleanup attempt {attempt + 1} failed, retrying...")
                        time.sleep(0.5)
                    else:
                        print(f"Warning: Could not fully cleanup {self.test_dir}: {e}")

    def _get_test_path(self, name):
        return os.path.join(self.test_dir, name)

    def _generate_random_vectors(self, n, dim):
        vecs = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_concurrent_reads(self):
        print("\n=== Concurrent Reads ===")

        index_path = self._get_test_path("concurrent_reads")
        dim = 128
        n_vectors = 10000

        db = self.VectorEngine(index_path, dim)
        db.init("build")

        vectors = self._generate_random_vectors(n_vectors, dim)
        for i, vec in enumerate(vectors):
            db.ingest(f"vec_{i}", vec)

        db.finalize()

        results = []
        errors = []

        def search_worker(thread_id, num_searches):
            try:
                local_db = self.VectorEngine(index_path, dim)
                for i in range(num_searches):
                    query = self._generate_random_vectors(1, dim)[0]
                    res = local_db.search(query, k=10)
                    assert len(res) > 0, f"Thread {thread_id}: Empty results"
                    assert len(res) <= 10, f"Thread {thread_id}: Too many results"
                results.append((thread_id, num_searches))
            except Exception as e:
                errors.append((thread_id, str(e)))

        n_threads = 10
        searches_per_thread = 500
        threads = []

        for i in range(n_threads):
            t = Thread(target=search_worker, args=(i, searches_per_thread))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert (
            len(results) == n_threads
        ), f"Not all threads completed: {len(results)}/{n_threads}"

        db.close()
        print(
            f"[OK] {n_threads} threads performed {n_threads * searches_per_thread} searches successfully"
        )

    def test_concurrent_writes_multiprocess(self):
        print("\n=== Concurrent Writes (Multiprocess) ===")

        index_path = self._get_test_path("concurrent_writes_mp")
        dim = 64

        db = self.VectorEngine(index_path, dim)
        db.init("build")
        for i in range(1000):
            vec = self._generate_random_vectors(1, dim)[0]
            db.ingest(f"initial_{i}", vec)
        db.finalize()
        db.close()

        def insert_worker(proc_id, n_inserts, result_queue):
            try:
                local_db = self.VectorEngine(index_path, dim)
                local_db.init("insert")

                for i in range(n_inserts):
                    vec = self._generate_random_vectors(1, dim)[0]
                    local_db.ingest(f"proc_{proc_id}_vec_{i}", vec)

                local_db.finalize()
                local_db.close()
                result_queue.put(("success", proc_id, n_inserts))
            except Exception as e:
                result_queue.put(("error", proc_id, str(e)))

        n_processes = 5
        inserts_per_process = 20
        result_queue = Queue()
        processes = []

        for i in range(n_processes):
            p = Thread(
                target=insert_worker, args=(i, inserts_per_process, result_queue)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, f"Errors in processes: {errors}"
        assert len(results) == n_processes, f"Not all processes completed"

        db = self.VectorEngine(index_path, dim)
        query = self._generate_random_vectors(1, dim)[0]
        res = db.search(query, k=200)
        assert len(res) > 100, f"Expected >100 results, got {len(res)}"
        db.close()

        print(
            f"[OK] {n_processes} processes inserted {n_processes * inserts_per_process} vectors"
        )

    def test_concurrent_deletes(self):
        print("\n=== Concurrent Deletes ===")

        index_path = self._get_test_path("concurrent_deletes")
        dim = 64
        n_vectors = 5000

        db = self.VectorEngine(index_path, dim)
        db.init("build")
        vectors = self._generate_random_vectors(n_vectors, dim)
        for i, vec in enumerate(vectors):
            db.ingest(f"vec_{i}", vec)
        db.finalize()
        db.close()

        def delete_worker(proc_id, ids_to_delete, result_queue):
            try:
                local_db = self.VectorEngine(index_path, dim)
                deleted_count, not_found = local_db.delete_items(
                    ids_to_delete, return_not_found=True
                )
                local_db.close()
                result_queue.put(
                    (
                        "success",
                        proc_id,
                        deleted_count,
                        len(not_found) if not_found else 0,
                    )
                )
            except Exception as e:
                result_queue.put(("error", proc_id, str(e)))

        n_processes = 4
        ids_per_process = 150
        result_queue = Queue()
        processes = []

        for i in range(n_processes):
            start_id = i * 100
            ids = [f"vec_{j}" for j in range(start_id, start_id + ids_per_process)]
            p = Thread(target=delete_worker, args=(i, ids, result_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, f"Errors in deletion: {errors}"

        total_deleted = sum(r[2] for r in results if r[0] == "success")
        print(
            f"[OK] {n_processes} processes attempted deletions, total deleted: {total_deleted}"
        )

    def test_mixed_read_write(self):
        print("\n=== Mixed Read/Write ===")

        index_path = self._get_test_path("mixed_rw")
        dim = 96

        db = self.VectorEngine(index_path, dim)
        db.init("build")
        for i in range(200):
            vec = self._generate_random_vectors(1, dim)[0]
            db.ingest(f"initial_{i}", vec)
        db.finalize()
        db.close()

        errors = []
        barrier = Barrier(6)

        def reader_worker(thread_id, n_searches):
            try:
                barrier.wait()
                local_db = self.VectorEngine(index_path, dim)
                for i in range(n_searches):
                    query = self._generate_random_vectors(1, dim)[0]
                    res = local_db.search(query, k=5)
                    time.sleep(0.01)
                local_db.close()
            except Exception as e:
                errors.append(("reader", thread_id, str(e)))

        def writer_worker(thread_id, n_inserts):
            try:
                barrier.wait()
                local_db = self.VectorEngine(index_path, dim)
                local_db.init("insert")
                for i in range(n_inserts):
                    vec = self._generate_random_vectors(1, dim)[0]
                    local_db.ingest(f"writer_{thread_id}_vec_{i}", vec)
                    time.sleep(0.02)
                local_db.finalize()
                local_db.close()
            except Exception as e:
                errors.append(("writer", thread_id, str(e)))

        threads = []

        for i in range(3):
            t = Thread(target=reader_worker, args=(i, 30))
            threads.append(t)
            t.start()

        for i in range(3):
            t = Thread(target=writer_worker, args=(i, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        print(f"[OK] 3 readers and 3 writers operated concurrently without errors")

    def test_concurrent_upserts(self):
        print("\n=== Concurrent Upserts ===")

        index_path = self._get_test_path("concurrent_upserts")
        dim = 128

        db = self.VectorEngine(index_path, dim)
        db.init("build")
        for i in range(1000):
            vec = self._generate_random_vectors(1, dim)[0]
            db.ingest(f"vec_{i}", vec)
        db.finalize()
        db.close()

        def upsert_worker(proc_id, result_queue):
            try:
                local_db = self.VectorEngine(index_path, dim)
                local_db.init("upsert")

                for i in range(50):
                    vec = self._generate_random_vectors(1, dim)[0]
                    local_db.ingest(f"vec_{i}", vec)

                local_db.finalize()
                local_db.close()
                result_queue.put(("success", proc_id))
            except Exception as e:
                result_queue.put(("error", proc_id, str(e)))
                traceback.print_exc()

        n_processes = 2
        result_queue = Queue()
        processes = []

        for i in range(n_processes):
            p = Thread(target=upsert_worker, args=(i, result_queue))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, f"Errors in upserts: {errors}"

        db = self.VectorEngine(index_path, dim)
        query = self._generate_random_vectors(1, dim)[0]
        res = db.search(query, k=10)
        assert len(res) > 0, "Index corrupted after concurrent upserts"
        db.close()

        print(
            f"[OK] {n_processes} processes performed overlapping upserts successfully"
        )

    def test_rebuild_with_readers(self):
        print("\n=== Rebuild with Active Readers ===")

        index_path = self._get_test_path("rebuild_readers")
        dim = 80

        db = self.VectorEngine(index_path, dim)
        db.init("build")
        for i in range(5000):
            vec = self._generate_random_vectors(1, dim)[0]
            db.ingest(f"vec_{i}", vec)
        db.finalize()
        db.close()

        errors = []
        rebuild_done = []

        def reader_worker(thread_id):
            try:
                local_db = self.VectorEngine(index_path, dim)
                for i in range(100):
                    query = self._generate_random_vectors(1, dim)[0]
                    res = local_db.search(query, k=10)
                    assert len(res) > 0, f"Thread {thread_id}: Got empty results"
                    time.sleep(0.01)
                local_db.close()
            except Exception as e:
                errors.append(("reader", thread_id, str(e)))

        def rebuilder_worker():
            try:
                time.sleep(0.2)
                local_db = self.VectorEngine(index_path, dim)
                local_db.rebuild_compact()
                local_db.close()
                rebuild_done.append(True)
            except Exception as e:
                errors.append(("rebuilder", str(e)))

        threads = []

        for i in range(10):
            t = Thread(target=reader_worker, args=(i,))
            threads.append(t)
            t.start()

        t = Thread(target=rebuilder_worker)
        threads.append(t)
        t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(rebuild_done) == 1, "Rebuild did not complete"
        print(f"[OK] Rebuild completed while 4 readers were active")

    def test_stress_many_operations(self):
        print("\n=== Stress Test ===")

        index_path = self._get_test_path("stress_test")
        dim = 128

        db = self.VectorEngine(index_path, dim)
        db.init("build")
        for i in range(50000):
            vec = self._generate_random_vectors(1, dim)[0]
            db.ingest(f"vec_{i}", vec)
        db.finalize()
        db.close()

        errors = []

        def operation_worker(worker_id, op_type):
            try:
                local_db = self.VectorEngine(index_path, dim)

                if op_type == "search":
                    for i in range(1000):
                        query = self._generate_random_vectors(1, dim)[0]
                        local_db.search(query, k=10)

                elif op_type == "insert":
                    local_db.init("insert")
                    for i in range(200):
                        vec = self._generate_random_vectors(1, dim)[0]
                        local_db.ingest(f"worker_{worker_id}_vec_{i}", vec)
                    local_db.finalize()

                elif op_type == "delete":
                    ids = [
                        f"vec_{i}" for i in range(worker_id * 10, worker_id * 10 + 10)
                    ]
                    local_db.delete_items(ids)

                local_db.close()
            except Exception as e:
                errors.append((op_type, worker_id, str(e)))

        threads = []

        # Mix of operations
        for i in range(10):
            t = Thread(target=operation_worker, args=(i, "search"))
            threads.append(t)

        for i in range(3):
            t = Thread(target=operation_worker, args=(i, "insert"))
            threads.append(t)

        for i in range(3):
            t = Thread(target=operation_worker, args=(i, "delete"))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        print(f"[OK] Stress test completed")

    def test_file_lock_contention(self):
        print("\n=== File Lock Contention ===")

        index_path = self._get_test_path("lock_contention")
        dim = 128
        db = self.VectorEngine(index_path, dim)
        db.init("build")
        for i in range(10000):
            vec = self._generate_random_vectors(1, dim)[0]
            db.ingest(f"vec_{i}", vec)
        db.finalize()
        db.close()

        def write_worker(proc_id, result_queue):
            try:
                local_db = self.VectorEngine(index_path, dim)

                for op_num in range(3):
                    local_db.init("insert")
                    vec = self._generate_random_vectors(1, dim)[0]
                    local_db.ingest(f"proc_{proc_id}_op_{op_num}", vec)
                    local_db.finalize()

                local_db.close()
                result_queue.put(("success", proc_id))
            except Exception as e:
                result_queue.put(("error", proc_id, str(e)))

        n_processes = 8
        result_queue = Queue()
        processes = []

        for i in range(n_processes):
            p = Thread(target=write_worker, args=(i, result_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, f"Errors under lock contention: {errors}"
        assert len(results) == n_processes, "Not all processes completed"

        print(f"[OK] {n_processes} processes competed for file locks successfully")

    def run_all_tests(self):
        print("\n" + "=" * 60)
        print("VectorEngine Concurrent Test Suite")
        print("=" * 60)

        self.setup()

        try:
            self.test_concurrent_reads()
            self.test_concurrent_writes_multiprocess()
            self.test_concurrent_deletes()
            self.test_mixed_read_write()
            self.test_concurrent_upserts()
            self.test_rebuild_with_readers()
            self.test_file_lock_contention()
            self.test_stress_many_operations()

            print("\n" + "=" * 60)
            print("[OK] All tests passed!")
            print("=" * 60)

        finally:
            self.teardown()


if __name__ == "__main__":
    from brinicle import VectorEngine

    tests = VectorEngineConcurrentTests(VectorEngine)
    tests.run_all_tests()
