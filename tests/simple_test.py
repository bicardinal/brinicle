import os
import shutil
import tempfile
import traceback

import numpy as np


class VectorEngineSimpleTests:

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

    def test_simple_insert(self):
        D = 2
        n = 5
        X = np.random.randn(n, D).astype(np.float32)
        Q = np.random.randn(D).astype(np.float32)
        engine = brinicle.VectorEngine(self._get_test_path("empty_id"), dim=D, delta_ratio=0.1)
        engine.init(mode="build")
        for eid in range(n):
            engine.ingest(str(eid), X[eid])
        engine.finalize()
        search = engine.search(Q, k=n)

        assert len(search) == n, "invalid search length"
        assert sorted(search) == [str(i) for i in range(n)], "invalid ids"
        assert len(set(search)) == n, "duplicate results" # to see if there are duplicates

        Y = np.random.randn(5, D).astype(np.float32)
        engine.init(mode="insert")
        for eid in range(5):
            engine.ingest(str(eid) + "x", Y[eid])
        engine.finalize()
        search = engine.search(Q, k=n + 5)
        print(search)
        assert len(search) == n + 5, "invalid search length"
        assert len(set(search)) == n + 5, "duplicate results" # to see if there are duplicates
        assert sorted(search) == sorted([str(i) for i in range(n)] + [str(i) + 'x' for i in range(5)]), "invalid ids" # to see if new inserts appear in the results.



if __name__ == "__main__":
    import brinicle

    tests = VectorEngineSimpleTests()
    tests.run_all()
