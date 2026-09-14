"""
Scratch checks for the raw distance kernels and brute-force kNN exposed by
the extension. Compares against numpy on random data.
"""

import sys
import time

import numpy as np

from brinicle import _brinicle


def check_distances():
    rng = np.random.default_rng(0)
    for dim in (1, 3, 8, 15, 16, 17, 384, 1024):
        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        diff = a - b
        np_l2 = float(np.dot(diff, diff))
        np_dot = float(np.dot(a, b))
        got_l2 = _brinicle.l2_sqr(a, b)
        got_dot = _brinicle.dot_product(a, b)
        assert np.isclose(got_l2, np_l2, rtol=1e-4, atol=1e-4), (dim, got_l2, np_l2)
        assert np.isclose(got_dot, np_dot, rtol=1e-4, atol=1e-4), (dim, got_dot, np_dot)
    print("[OK] l2_sqr / dot_product match numpy")


def check_brute_knn():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 64)).astype(np.float32)
    Q = rng.standard_normal((16, 64)).astype(np.float32)
    k = 10

    t0 = time.perf_counter()
    idx, dist = _brinicle.brute_knn_batch(X, Q, k=k, n_jobs=2)
    print(f"brute_knn_batch: {(time.perf_counter() - t0) * 1000:.2f} ms")

    assert idx.shape == (16, k) and dist.shape == (16, k)

    full = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    expected = np.argsort(full, axis=1)[:, :k]
    for qi in range(Q.shape[0]):
        assert set(idx[qi].tolist()) == set(expected[qi].tolist()), qi
        assert np.all(np.diff(dist[qi]) >= -1e-6), "distances not sorted"
    print("[OK] brute_knn_batch matches numpy argsort")


if __name__ == "__main__":
    check_distances()
    check_brute_knn()
    sys.exit(0)
