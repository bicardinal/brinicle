import os
import shutil
import tempfile
import traceback

import numpy as np


class VectorEngineBatchCorrectnessTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="vectorengine_test_")
        print(f"Test directory: {self.test_dir}")

    def teardown(self):
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up: {self.test_dir}")

    def _get_test_path(self, name):
        return os.path.join(self.test_dir, f"test_idx_{name}")

    def _run_test(self, test_method):
        self.test_count += 1
        test_name = test_method.__name__
        try:
            print(f"\nRunning: {test_name}")
            test_method()
            self.passed_count += 1
            print(f"[OK] {test_name} PASSED")
        except Exception as e:
            print(f"[NOT OK] {test_name} FAILED: {e}")
            traceback.print_exc()

    def run_all(self):
        self.setup()
        try:
            test_methods = [
                getattr(self, m)
                for m in dir(self)
                if m.startswith("test_") and callable(getattr(self, m))
            ]
            for test_method in test_methods:
                self._run_test(test_method)
            print(f"\n{'='*60}")
            print(f"Results: {self.passed_count}/{self.test_count} tests passed")
            print(f"{'='*60}")
        finally:
            self.teardown()

    def test_single_query_batch_matches_search(self):
        D = 2
        n = 10000
        k = n
        X = np.random.randn(n, D).astype(np.float32)
        Q = np.random.randn(D).astype(np.float32)
        engine = brinicle.VectorEngine(
            self._get_test_path("single_batch"), dim=D, delta_ratio=0.1
        )
        engine.init(mode="build")
        for eid in range(n):
            engine.ingest(str(eid), X[eid])
        engine.finalize()

        search = [int(x) for x in engine.search(Q, k=k)]
        batch = engine.search_batch(Q.reshape(1, -1), k=k, n_jobs=1)
        assert isinstance(batch, list) and len(batch) == 1, "batch must be one row per query"
        search_batch = [int(x) for x in batch[0]]

        assert len(search) == k, "invalid search length"
        assert sorted(search) == [i for i in range(k)], "invalid ids"
        assert len(set(search)) == k, "duplicate results"
        assert search_batch == search, "batch row differs from single search"
        engine.close()

    def test_multi_query_batch_matches_search(self):
        D = 16
        n = 5000
        nq = 32
        k = 25
        X = np.random.randn(n, D).astype(np.float32)
        Q = np.random.randn(nq, D).astype(np.float32)
        engine = brinicle.VectorEngine(
            self._get_test_path("multi_batch"), dim=D, delta_ratio=0.1
        )
        engine.init(mode="build")
        for eid in range(n):
            engine.ingest(str(eid), X[eid])
        engine.finalize()

        for n_jobs in (1, 4):
            batch = engine.search_batch(Q, k=k, n_jobs=n_jobs)
            assert len(batch) == nq, f"expected {nq} rows, got {len(batch)}"
            for qi in range(nq):
                single = engine.search(Q[qi], k=k)
                assert len(batch[qi]) == k, "invalid row length"
                assert batch[qi] == single, f"row {qi} differs (n_jobs={n_jobs})"

        # exact ground truth: batch top-1 must be the brute-force nearest
        full = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        nearest = full.argmin(axis=1)
        batch = engine.search_batch(Q, k=k, n_jobs=2)
        hits = sum(int(batch[qi][0]) == int(nearest[qi]) for qi in range(nq))
        assert hits >= nq * 0.9, f"top-1 recall too low: {hits}/{nq}"
        engine.close()

    def test_batch_with_distance_consistency(self):
        D = 8
        n = 3000
        k = 10
        X = np.random.randn(n, D).astype(np.float32)
        Q = np.random.randn(4, D).astype(np.float32)
        engine = brinicle.VectorEngine(
            self._get_test_path("batch_dist"), dim=D, delta_ratio=0.1
        )
        engine.init(mode="build")
        for eid in range(n):
            engine.ingest(str(eid), X[eid])
        engine.finalize()

        batch = engine.search_batch(Q, k=k)
        for qi in range(Q.shape[0]):
            with_dist = engine.search_with_distance(Q[qi], k=k)
            ids = [i for i, _ in with_dist]
            dists = [d for _, d in with_dist]
            assert ids == batch[qi], "search_with_distance ids differ from batch row"
            assert dists == sorted(dists), "distances not ascending"
            for i, d in with_dist:
                expected = float(((X[int(i)] - Q[qi]) ** 2).sum())
                assert abs(d - expected) < 1e-3 * max(1.0, expected), (i, d, expected)
        engine.close()


if __name__ == "__main__":
    import brinicle

    tests = VectorEngineBatchCorrectnessTests()
    tests.run_all()
