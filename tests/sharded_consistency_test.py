import os
import shutil
import tempfile
import threading
import traceback

import numpy as np


class ShardedConsistencyTests:
    """Empty shards, one engine shared across threads, and several engines
    over one path."""

    def __init__(self, VectorEngine):
        self.VectorEngine = VectorEngine
        self.test_dir = None
        self.test_count = 0
        self.passed_count = 0

    def setup(self):
        self.test_dir = tempfile.mkdtemp(prefix="sharded_consistency_test_")
        print(f"Test directory: {self.test_dir}")

    def teardown(self):
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up: {self.test_dir}")

    def _path(self, name):
        return os.path.join(self.test_dir, name)

    def _vectors(self, n, dim, seed=0):
        v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _run_test(self, test_method):
        self.test_count += 1
        name = test_method.__name__
        try:
            print(f"\nRunning: {name}")
            test_method()
            self.passed_count += 1
            print(f"[OK] {name} PASSED")
        except Exception as e:
            print(f"[NOT OK] {name} FAILED: {e}")
            traceback.print_exc()

    def run_all(self):
        self.setup()
        try:
            for name in sorted(dir(self)):
                if name.startswith("test_") and callable(getattr(self, name)):
                    self._run_test(getattr(self, name))
        finally:
            self.teardown()
        print(f"\n{self.passed_count}/{self.test_count} tests passed")
        return self.passed_count == self.test_count

    def test_writes_into_shards_empty_after_build(self):
        dim, n_shards = 16, 8
        db = self.VectorEngine(self._path("empty"), dim, n_shards=n_shards)
        vecs = self._vectors(300, dim)

        # Two items leave most of the eight shards without a segment.
        db.init("build")
        for i in range(2):
            db.ingest(f"v{i}", vecs[i])
        db.finalize()

        # Each mode must be able to create a shard's first segment.
        db.init("insert")
        for i in range(2, 100):
            db.ingest(f"v{i}", vecs[i])
        db.finalize()

        db.init("upsert")
        for i in range(100, 300):
            db.ingest(f"v{i}", vecs[i])
        db.finalize()

        for i in (0, 50, 250):
            res = db.search(vecs[i], k=1, efs=64)
            assert res == [f"v{i}"], f"v{i} not found after writes into empty shards: {res}"

        # A delete routed to a shard that has no segment reports not-found
        # instead of failing.
        fresh = self.VectorEngine(self._path("fresh"), dim, n_shards=n_shards)
        fresh.init("build")
        fresh.ingest("only", vecs[0])
        fresh.finalize()
        ids = [f"missing{i}" for i in range(40)]
        deleted, not_found = fresh.delete_items(ids, return_not_found=True)
        assert deleted == 0, "nothing should be deleted"
        assert sorted(not_found) == sorted(ids), "every id should be reported not found"

        db.close()
        fresh.close()

    def test_shared_engine_searches_during_writes(self):
        dim, n_shards = 32, 4
        db = self.VectorEngine(
            self._path("shared"), dim, M=16, ef_construction=100, n_shards=n_shards
        )
        vecs = self._vectors(6000, dim, seed=1)

        db.init("build")
        for i in range(3000):
            db.ingest(f"v{i}", vecs[i])
        db.finalize()

        stop = threading.Event()
        errors = []
        searches = [0]

        def reader():
            queries = self._vectors(20, dim, seed=2)
            while not stop.is_set():
                try:
                    for q in queries:
                        if not db.search(q, k=10, efs=64):
                            errors.append("empty result")
                        searches[0] += 1
                except Exception as e:
                    errors.append(repr(e))

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        try:
            for rnd in range(30):
                db.init("upsert")
                for i in range(100):
                    j = 3000 + (rnd * 100 + i) % 3000
                    db.ingest(f"v{j}", vecs[j])
                db.finalize(optimize=True)
                db.delete_items([f"v{rnd * 50 + i}" for i in range(50)])
        finally:
            stop.set()
            for t in threads:
                t.join()

        assert not errors, f"errors during concurrent search: {errors[:5]}"
        assert searches[0] > 0, "readers never ran"
        db.close()

    def test_separate_engines_see_each_others_writes(self):
        dim, n_shards = 16, 4
        path = self._path("separate")
        vecs = self._vectors(210, dim, seed=3)

        writer = self.VectorEngine(path, dim, n_shards=n_shards)
        writer.init("build")
        for i in range(200):
            writer.ingest(f"v{i}", vecs[i])
        writer.finalize()

        reader = self.VectorEngine(path, dim, n_shards=n_shards)

        # Several writes in quick succession, well inside one second of each
        # other, each of which the reader must pick up.
        for i in range(200, 210):
            writer.init("upsert")
            writer.ingest(f"v{i}", vecs[i])
            writer.finalize()
            res = reader.search(vecs[i], k=1, efs=64)
            assert res == [f"v{i}"], f"reader missed v{i}: {res}"

            writer.delete_items([f"v{i}"])
            res = reader.search(vecs[i], k=1, efs=64)
            assert res != [f"v{i}"], f"reader still returns deleted v{i}"

        writer.close()
        reader.close()


if __name__ == "__main__":
    import sys

    import brinicle

    ok = ShardedConsistencyTests(brinicle.VectorEngine).run_all()
    sys.exit(0 if ok else 1)
