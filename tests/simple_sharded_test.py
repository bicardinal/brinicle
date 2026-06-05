import os
import shutil
import tempfile
import traceback

import numpy as np


class VectorEngineShardingSimpleTests:

    def __init__(self):
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="vector_sharding_test_")
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

            print(f"\n{'=' * 60}")
            print(f"Results: {self.passed_count}/{self.test_count} tests passed")
            print(f"{'=' * 60}")

        finally:
            self.teardown()

    def test_sharded_simple_build_returns_all_ids(self):
        rng = np.random.default_rng(42)

        D = 64
        n = 1000
        n_shards = 10

        X = rng.standard_normal((n, D)).astype(np.float32)
        q = rng.standard_normal(D).astype(np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("simple_build_all_ids"),
            dim=D,
            n_shards=n_shards,
            M=32,
            ef_construction=300,
            ef_search=128,
            delta_ratio=0.1,
        )

        assert engine.n_shards == n_shards, "invalid n_shards property"

        engine.init(mode="build")

        for i in range(n):
            engine.ingest(str(i), X[i])

        engine.finalize()

        results = engine.search(q, k=n, efs=n)

        assert len(results) == n, "invalid result count"
        assert len(set(results)) == n, "duplicate results"
        assert sorted(results, key=int) == [str(i) for i in range(n)], "invalid ids"

        engine.close()

    def test_sharded_exact_vector_top1(self):
        rng = np.random.default_rng(123)

        D = 64
        n = 1000
        target = 777

        X = rng.standard_normal((n, D)).astype(np.float32)
        q = X[target].copy()

        engine = brinicle.VectorEngine(
            self._get_test_path("exact_vector_top1"),
            dim=D,
            n_shards=10,
            M=32,
            ef_construction=300,
            ef_search=128,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(n):
            engine.ingest(str(i), X[i])

        engine.finalize()

        result = engine.search(q, k=1, efs=256)

        assert len(result) == 1, "invalid top1 result length"
        assert result[0] == str(target), "exact vector should be top1"

        result_with_distance = engine.search_with_distance(q, k=1, efs=256)

        assert len(result_with_distance) == 1, "invalid top1 distance result length"
        assert result_with_distance[0][0] == str(target), "exact vector should be top1 with distance"
        assert abs(result_with_distance[0][1]) < 1e-5, "exact vector distance should be near zero"

        engine.close()

    def test_sharded_search_with_distance_sorted(self):
        rng = np.random.default_rng(7)

        D = 32
        n = 500

        X = rng.standard_normal((n, D)).astype(np.float32)
        q = rng.standard_normal(D).astype(np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("distance_sorted"),
            dim=D,
            n_shards=8,
            M=24,
            ef_construction=250,
            ef_search=128,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(n):
            engine.ingest(str(i), X[i])

        engine.finalize()

        results = engine.search_with_distance(q, k=20, efs=128)

        assert len(results) == 20, "invalid search_with_distance length"
        assert len(set(r[0] for r in results)) == 20, "duplicate ids in distance results"

        distances = [r[1] for r in results]

        assert distances == sorted(distances), "distances are not sorted"

        engine.close()

    def test_sharded_insert(self):
        rng = np.random.default_rng(88)

        D = 64
        base_n = 300
        insert_n = 20

        X = rng.standard_normal((base_n, D)).astype(np.float32)
        inserted = rng.standard_normal((insert_n, D)).astype(np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("insert"),
            dim=D,
            n_shards=10,
            M=32,
            ef_construction=300,
            ef_search=128,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(base_n):
            engine.ingest(str(i), X[i])

        engine.finalize()

        engine.init(mode="insert")

        for i in range(insert_n):
            engine.ingest(f"inserted_{i}", inserted[i])

        engine.finalize()

        q = inserted[5].copy()

        top = engine.search(q, k=1, efs=256)

        assert top == ["inserted_5"], "inserted exact vector should be top1"

        all_results = engine.search(q, k=base_n + insert_n, efs=base_n + insert_n)

        expected_ids = [str(i) for i in range(base_n)] + [f"inserted_{i}" for i in range(insert_n)]

        assert len(all_results) == base_n + insert_n, "invalid result count after insert"
        assert len(set(all_results)) == base_n + insert_n, "duplicate results after insert"
        assert sorted(all_results) == sorted(expected_ids), "inserted ids not found"

        engine.close()

    def test_sharded_upsert_same_id_moves_to_new_vector(self):
        D = 8

        old_vec = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        other_vec = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_vec = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("upsert"),
            dim=D,
            n_shards=6,
            M=16,
            ef_construction=200,
            ef_search=64,
            delta_ratio=0.1,
        )

        engine.init(mode="build")
        engine.ingest("p1", old_vec)
        engine.ingest("p2", other_vec)
        engine.finalize()

        before = engine.search(old_vec, k=1, efs=64)

        assert before == ["p1"], "p1 should match old vector before upsert"

        engine.init(mode="upsert")
        engine.ingest("p1", new_vec)
        engine.finalize()

        after_new = engine.search(new_vec, k=1, efs=64)

        assert after_new == ["p1"], "p1 should match new vector after upsert"

        all_results = engine.search(new_vec, k=2, efs=64)

        assert len(all_results) == 2, "invalid total result count after upsert"
        assert len(set(all_results)) == 2, "duplicate results after upsert"
        assert sorted(all_results) == ["p1", "p2"], "upsert changed ids incorrectly"

        engine.close()

    def test_sharded_delete(self):
        rng = np.random.default_rng(55)

        D = 32
        n = 100

        X = rng.standard_normal((n, D)).astype(np.float32)
        q = rng.standard_normal(D).astype(np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("delete"),
            dim=D,
            n_shards=10,
            M=24,
            ef_construction=250,
            ef_search=128,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(n):
            engine.ingest(str(i), X[i])

        engine.finalize()

        deleted_count, not_found = engine.delete_items(
            ["1", "4", "17", "missing"],
            return_not_found=True,
        )

        assert deleted_count == 3, "invalid delete count"
        assert set(not_found) == {"missing"}, "invalid not_found list"

        results = engine.search(q, k=n, efs=n)

        assert "1" not in results, "deleted id 1 still appears"
        assert "4" not in results, "deleted id 4 still appears"
        assert "17" not in results, "deleted id 17 still appears"
        assert len(results) == n - 3, "invalid result count after delete"
        assert len(set(results)) == n - 3, "duplicate results after delete"

        expected_ids = [str(i) for i in range(n) if i not in (1, 4, 17)]

        assert sorted(results, key=int) == expected_ids, "invalid ids after delete"

        engine.close()

    def test_sharded_rebuild_and_optimize(self):
        rng = np.random.default_rng(91)

        D = 32
        n = 200

        X = rng.standard_normal((n, D)).astype(np.float32)
        q = rng.standard_normal(D).astype(np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("rebuild_optimize"),
            dim=D,
            n_shards=10,
            M=24,
            ef_construction=250,
            ef_search=128,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(n):
            engine.ingest(str(i), X[i])

        engine.finalize()

        engine.delete_items(["1", "2", "3"])

        before = engine.search(q, k=n, efs=n)

        assert len(before) == n - 3, "invalid result count before rebuild"
        assert "1" not in before, "deleted id 1 appears before rebuild"
        assert "2" not in before, "deleted id 2 appears before rebuild"
        assert "3" not in before, "deleted id 3 appears before rebuild"

        engine.rebuild_compact(
            M=24,
            ef_construction=250,
            ef_search=128,
            build_n_threads=1,
            seed=0,
        )

        engine.optimize_graph()

        after = engine.search(q, k=n, efs=n)

        assert len(after) == n - 3, "invalid result count after rebuild"
        assert len(set(after)) == n - 3, "duplicate results after rebuild"
        assert "1" not in after, "deleted id 1 appears after rebuild"
        assert "2" not in after, "deleted id 2 appears after rebuild"
        assert "3" not in after, "deleted id 3 appears after rebuild"

        expected_ids = [str(i) for i in range(n) if i not in (1, 2, 3)]

        assert sorted(after, key=int) == expected_ids, "invalid ids after rebuild"

        engine.close()

    def test_sharded_search_batch(self):
        D = 16

        X = np.array(
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        Q = X.copy()

        engine = brinicle.VectorEngine(
            self._get_test_path("search_batch"),
            dim=D,
            n_shards=4,
            M=16,
            ef_construction=200,
            ef_search=64,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(len(X)):
            engine.ingest(f"v{i}", X[i])

        engine.finalize()

        results = engine.search_batch(Q, k=1, efs=64, n_jobs=2)

        assert len(results) == 4, "invalid batch result length"
        assert results[0] == ["v0"], "invalid batch result 0"
        assert results[1] == ["v1"], "invalid batch result 1"
        assert results[2] == ["v2"], "invalid batch result 2"
        assert results[3] == ["v3"], "invalid batch result 3"

        engine.close()

    def test_sharded_empty_batch(self):
        D = 16

        Q = np.empty((0, D), dtype=np.float32)

        engine = brinicle.VectorEngine(
            self._get_test_path("empty_batch"),
            dim=D,
            n_shards=5,
        )

        results = engine.search_batch(Q, k=10)

        assert results == [], "empty batch should return empty list"

        engine.close()

    def test_sharded_reopen_existing_index(self):
        rng = np.random.default_rng(131)

        D = 32
        n = 300
        target = 123
        path = self._get_test_path("reopen")

        X = rng.standard_normal((n, D)).astype(np.float32)

        engine = brinicle.VectorEngine(
            path,
            dim=D,
            n_shards=10,
            M=24,
            ef_construction=250,
            ef_search=128,
            delta_ratio=0.1,
        )

        engine.init(mode="build")

        for i in range(n):
            engine.ingest(str(i), X[i])

        engine.finalize()
        engine.close()

        reopened = brinicle.VectorEngine(
            path,
            dim=D,
            n_shards=10,
        )

        result = reopened.search(X[target], k=1, efs=256)

        assert result == [str(target)], "reopened sharded index failed exact top1 search"
        assert reopened.n_shards == 10, "reopened index has invalid shard count"

        reopened.close()

    def test_sharded_n_shards_mismatch_should_fail(self):
        rng = np.random.default_rng(222)

        D = 16
        path = self._get_test_path("n_shards_mismatch")

        X = rng.standard_normal((20, D)).astype(np.float32)

        engine = brinicle.VectorEngine(
            path,
            dim=D,
            n_shards=5,
        )

        engine.init(mode="build")

        for i in range(20):
            engine.ingest(str(i), X[i])

        engine.finalize()
        engine.close()

        try:
            brinicle.VectorEngine(
                path,
                dim=D,
                n_shards=10,
            )
            assert False, "opening with wrong n_shards should fail"
        except RuntimeError as e:
            assert "n_shards mismatch" in str(e), "invalid n_shards mismatch error message"

    def test_sharded_dimension_validation(self):
        D = 16

        engine = brinicle.VectorEngine(
            self._get_test_path("dimension_validation"),
            dim=D,
            n_shards=4,
        )

        bad_vec = np.random.randn(D + 1).astype(np.float32)
        good_vec = np.random.randn(D).astype(np.float32)

        engine.init(mode="build")

        try:
            engine.ingest("bad", bad_vec)
            assert False, "ingest dimension mismatch should raise RuntimeError"
        except RuntimeError as e:
            assert "dimension mismatch" in str(e), "invalid ingest dimension error message"

        engine.ingest("good", good_vec)
        engine.finalize()

        bad_q = np.random.randn(D + 1).astype(np.float32)

        try:
            engine.search(bad_q, k=1)
            assert False, "search dimension mismatch should raise RuntimeError"
        except RuntimeError as e:
            assert "dimension mismatch" in str(e), "invalid search dimension error message"

        engine.close()


if __name__ == "__main__":
    import brinicle

    tests = VectorEngineShardingSimpleTests()
    tests.run_all()
