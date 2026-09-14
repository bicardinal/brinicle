"""
Scratch lifecycle walk-through for VectorEngine: build, repeated upsert
across engine reopen, delete, rebuild. Prints results at each step.
"""

import shutil
import sys
import tempfile

import numpy as np

import brinicle

D = 2
n = 1000
rng = np.random.default_rng(0)
X = rng.standard_normal((n, D)).astype(np.float32)
Q = rng.standard_normal(D).astype(np.float32)

tmp = tempfile.mkdtemp(prefix="debug_tests_")
path = f"{tmp}/test_database"

try:
    try:
        brinicle.VectorEngine(path, dim=D, delta_ratio=0.6)
        raise AssertionError("delta_ratio > 0.5 should be rejected")
    except ValueError:
        pass

    db = brinicle.VectorEngine(path, dim=D, M=16, ef_construction=200, ef_search=64, seed=123, delta_ratio=0.09)

    db.init(mode="build")
    for eid in range(n):
        db.ingest(str(eid), X[eid])
    db.finalize()

    print("BUILD", db.search(Q, k=10))
    print("rebuild?", db.needs_rebuild())
    db.close()

    for i in range(10):
        db = brinicle.VectorEngine(path, dim=D)
        Y = rng.standard_normal((5, D)).astype(np.float32)
        db.init(mode="upsert")
        for eid in range(2):
            db.ingest(str(eid), Y[eid])
        db.finalize()
        print(f"UPSERT {i}", db.search(Q, k=20))
        db.close()

    db = brinicle.VectorEngine(path, dim=D)
    print("DELETE", db.delete_items(["0", "1", "missing"], return_not_found=True))
    res = db.search(Q, k=n)
    assert "0" not in res and "1" not in res, "deleted ids still returned"

    print("rebuild?", db.needs_rebuild())
    db.rebuild_compact(M=16, ef_construction=200, ef_search=64)
    res = db.search(Q, k=n)
    assert len(res) == n - 2, len(res)
    print("REBUILD ok, active =", len(res))
    db.destroy()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

sys.exit(0)
