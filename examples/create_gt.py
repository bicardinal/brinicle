import numpy as np

HEADER_SIZE = 5
ENCODING_VERSION = 1

W_TITLE = 0.55
W_ATTR  = 0.15
W_BRAND = 0.20
W_CAT   = 0.10


def _as_id(x: float) -> int:
    return int(x) if x > 0 else 0


def _parse_vec(v: np.ndarray):
    if v.shape[0] < HEADER_SIZE:
        raise ValueError("vector dim is smaller than header size")

    ver = _as_id(v[0])
    if ver != ENCODING_VERSION:
        raise ValueError(f"unexpected encoding version: {ver}")

    payload = v.shape[0] - HEADER_SIZE

    title_n = min(int(v[1]), payload)
    attr_n = min(int(v[2]), payload - title_n)

    brand_id = _as_id(v[3])
    cat_id = _as_id(v[4])

    title = v[HEADER_SIZE:HEADER_SIZE + title_n].astype(np.int64, copy=False)
    attr = v[HEADER_SIZE + title_n:HEADER_SIZE + title_n + attr_n].astype(np.int64, copy=False)

    return title, attr, brand_id, cat_id


def _jaccard_distance_sorted_ids(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 0.0

    inter = np.intersect1d(a, b, assume_unique=True).size
    uni = len(a) + len(b) - inter
    if uni == 0:
        return 0.0
    return 1.0 - (inter / uni)


def stage1_distance_from_parts(q_parts, x_parts) -> float:
    q_title, q_attr, q_brand, q_cat = q_parts
    x_title, x_attr, x_brand, x_cat = x_parts

    title_dist = _jaccard_distance_sorted_ids(q_title, x_title)
    attr_dist = _jaccard_distance_sorted_ids(q_attr, x_attr)

    # 0 means unspecified -> no penalty
    brand_dist = 0.0 if (q_brand == 0 or x_brand == 0 or q_brand == x_brand) else 1.0
    cat_dist   = 0.0 if (q_cat   == 0 or x_cat   == 0 or q_cat   == x_cat)   else 1.0

    return (
        W_TITLE * title_dist +
        W_ATTR  * attr_dist +
        W_BRAND * brand_dist +
        W_CAT   * cat_dist
    )


def build_bruteforce_gt(X: np.ndarray, Q: np.ndarray, gt_k: int = 100) -> np.ndarray:

    n_products = X.shape[0]
    n_queries = Q.shape[0]
    gt_k = min(gt_k, n_products)

    parsed_X = [_parse_vec(X[i]) for i in range(n_products)]
    parsed_Q = [_parse_vec(Q[i]) for i in range(n_queries)]

    GT = np.empty((n_queries, gt_k), dtype=np.int64)

    for i, q_parts in enumerate(parsed_Q):
        dists = np.empty(n_products, dtype=np.float32)

        for j, x_parts in enumerate(parsed_X):
            dists[j] = stage1_distance_from_parts(q_parts, x_parts)

        # partial top-k
        top_idx = np.argpartition(dists, gt_k - 1)[:gt_k]

        # deterministic ordering: first by distance, then by index
        order = np.lexsort((top_idx, dists[top_idx]))
        GT[i] = top_idx[order]

        print(f"GT {i+1}/{n_queries}", end="\r")

    print()
    return GT

