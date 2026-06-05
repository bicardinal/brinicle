# Vector Engine

`VectorEngine` is brinicle's low-level engine for raw vector similarity search.

Use it when you already have embeddings or numeric vectors and want disk-first HNSW approximate nearest-neighbor search without item-level lexical encoding.

It supports:

* raw vector search
* single-query search
* batch search
* search with distances
* local sharding for large indexes
* insert, upsert, and delete
* compact rebuild and graph optimization
* optional build-time threading

`VectorEngine` is the lowest-level public engine in brinicle. Higher-level engines such as `ItemSearchEngine` and `AutocompleteEngine` use the same disk-first HNSW infrastructure, but encode their inputs before indexing and searching.

---

## Constructor

```python
engine = brinicle.VectorEngine(
    index_path,
    dim=0,
    delta_ratio=0.10,
    M=16,
    ef_construction=200,
    ef_search=64,
    build_n_threads=1,
    seed=0,
    dist_func="l2",
    lexical_config=None,
    autocomplete_config=None,
    n_shards=1,
)
```

### Parameters

| Parameter             | Meaning                                                     |
| --------------------- | ----------------------------------------------------------- |
| `index_path`          | Base path for the index files                               |
| `dim`                 | Vector dimension. Use `0` only when loading an existing index |
| `delta_ratio`         | Maintenance threshold for delta and deleted records         |
| `M`                   | HNSW graph connectivity                                     |
| `ef_construction`     | Build-time search width                                     |
| `ef_search`           | Default query-time search width                             |
| `build_n_threads`     | Number of build threads                                     |
| `seed`                | Random seed for graph construction                          |
| `dist_func`           | Distance function name. Use `"l2"` for normal vector search |
| `lexical_config`      | Advanced/internal scoring config used by higher-level engines |
| `autocomplete_config` | Advanced/internal scoring config used by higher-level engines |
| `n_shards`            | Number of local shards used when building the index         |

For normal vector search, leave `dist_func`, `lexical_config`, and `autocomplete_config` at their defaults.

Example:

```python
engine = brinicle.VectorEngine(
    "vector_index",
    dim=384,
    M=48,
    ef_construction=1024,
    ef_search=512,
    build_n_threads=4,
    delta_ratio=0.1,
)
```

---

## Local Sharding

`n_shards` controls how many local shards brinicle creates for the index during build.

The current sharding implementation is local sharding. It is not distributed sharding across multiple machines.

During insertion, brinicle uses hash-based routing to decide which shard receives each vector. During search, brinicle searches all shards, then merges the partial results into a final ranked result list.

For multi-shard indexes, search methods accept `n_jobs`, which controls how many shards are searched in parallel.

Higher `n_jobs` can reduce search latency, but it also increases CPU and I/O usage.

Example:

```python
engine = brinicle.VectorEngine(
    "large_vector_index",
    dim=384,
    M=48,
    ef_construction=1024,
    ef_search=512,
    build_n_threads=8,
    n_shards=50,
)

results = engine.search(
    query_vector,
    k=10,
    n_jobs=8,
)
```

As a practical starting point, keep `n_shards=1` for smaller indexes. Sharding becomes more useful when the index contains more than about 2 million elements. For example, for an index with around 100 million elements, `n_shards=50` can be a reasonable starting point.

Benchmark `n_shards` and `n_jobs` with your real data and hardware, because the best values depend on vector count, vector dimension, storage speed, and available CPU cores.

---

## Vector Format

Vectors must be one-dimensional `float32` arrays during ingest and search.

```python
import numpy as np

vec = np.random.randn(384).astype("float32")
```

For batch search, queries must be a two-dimensional `float32` array:

```python
queries = np.random.randn(100, 384).astype("float32")
```

Each query vector must have the same dimension as the index.

---

## Distance Functions

`VectorEngine` supports these distance functions:

| `dist_func`              | Meaning                       |
| ------------------------ | ----------------------------- |
| `"l2"`                   | Squared Euclidean distance    |
| `"cosine_distance"`      | `1 - cosine_similarity(a, b)` |
| `"dot_product_distance"` | `-dot_product(a, b)`          |

brinicle ranks results by ascending distance.

Smaller distance means a better match.

For `dot_product_distance`, a larger dot product becomes a smaller distance:

```text
dot_product = 0.90  -> distance = -0.90
dot_product = 0.20  -> distance = -0.20
```

So the result with distance `-0.90` is ranked before the result with distance `-0.20`.

---

## Basic Build Example

```python
import numpy as np
import brinicle

dim = 384
n = 1000

X = np.random.randn(n, dim).astype("float32")
Q = np.random.randn(dim).astype("float32")

engine = brinicle.VectorEngine(
    "vector_index",
    dim=dim,
    M=48,
    ef_construction=1024,
    ef_search=512,
)

engine.init(mode="build")

for i in range(n):
    engine.ingest(
        external_id=str(i),
        vec=X[i],
    )

engine.finalize()

results = engine.search(Q, k=10)

print(results)
```

`search(...)` returns external IDs:

```python
["37", "911", "104"]
```

---

## Search

Use `search(...)` to return external IDs only.

```python
results = engine.search(
    q,
    k=10,
    efs=64,
    threshold=float("inf"),
    n_jobs=1,
)
```

Example output:

```python
["42", "18", "901"]
```

### Parameters

| Parameter   | Meaning                                                       |
| ----------- | ------------------------------------------------------------- |
| `q`         | Query vector as a one-dimensional `float32` array              |
| `k`         | Maximum number of results                                     |
| `efs`       | Query-time search width                                       |
| `threshold` | Maximum accepted distance                                     |
| `n_jobs`    | Number of shards to search in parallel on multi-shard indexes |

For `n_shards=1`, you can usually ignore `n_jobs`.

Example:

```python
results = engine.search(
    Q,
    k=10,
    efs=128,
)

print(results)
```

---

## Search with Distance

Use `search_with_distance(...)` to return IDs and distances.

```python
results = engine.search_with_distance(
    Q,
    k=10,
    efs=128,
)
print(results)
```

Example output:

```python
[("37", 0.18), ("911", 0.21), ("104", 0.27)]
```

Smaller distance means a better match.

On multi-shard indexes, use `n_jobs` to control how many shards are searched in parallel if your installed version exposes it for this method.

---

## Batch Search

Use `search_batch(...)` to search multiple query vectors.

```python
results = engine.search_batch(
    Qs,
    k=10,
    efs=128,
    threshold=float("inf"),
    n_jobs=4,
)
```

`Qs` must be a two-dimensional `float32` array with shape:

```text
(number_of_queries, dim)
```

Example:

```python
queries = np.random.randn(32, dim).astype("float32")

results = engine.search_batch(
    queries,
    k=10,
    efs=128,
    n_jobs=4,
)

print(results)
```

Example output:

```python
[
    ["37", "911", "104"],
    ["12", "73", "88"],
]
```

`n_jobs` controls query-level parallelism and, on multi-shard indexes, shard-level parallelism. Higher values can reduce latency, but they also consume more CPU and I/O.

---

## Insert

Use `insert` mode to add new vectors to an existing index.

```python
new_vectors = np.random.randn(100, dim).astype("float32")

engine.init(mode="insert")

for i in range(100):
    engine.ingest(
        external_id=f"new_{i}",
        vec=new_vectors[i],
    )

engine.finalize()
```

Inserted records are added through the delta index. This allows brinicle to accept updates without rebuilding the full main index after every insert.

---

## Upsert

Use `upsert` mode to replace existing vectors or insert new ones.

```python
replacement_vectors = np.random.randn(100, dim).astype("float32")

engine.init(mode="upsert")

for i in range(100):
    engine.ingest(
        external_id=str(i),
        vec=replacement_vectors[i],
    )

engine.finalize()
```

If the external ID already exists, brinicle marks the old record as deleted and inserts the new version.

If the external ID does not exist, the vector is inserted as a new record.

---

## Delete

Use `delete_items(...)` to delete vectors by external ID.

```python
deleted_count, not_found = engine.delete_items(
    ["37", "911"],
    return_not_found=True,
)

print(deleted_count)
print(not_found)
```

If `return_not_found=False`, the second returned value is `None`.

Deletes are logical until compact rebuild.

---

## Rebuild and Optimize

Vector indexes use the same maintenance model as the higher-level engines.

```python
engine.needs_rebuild()
```

Returns whether the index has enough update or delete drift to justify rebuilding.

Use `rebuild_compact(...)` to rebuild the index from alive records.

```python
engine.rebuild_compact()
```

This:

* removes deleted records physically
* merges alive records from the main and delta indexes
* builds a new main index
* clears the delta index

You can also pass build parameters:

```python
engine.rebuild_compact(
    M=48,
    ef_construction=1024,
    ef_search=512,
    build_n_threads=4,
)
```

Use `optimize_graph(...)` to run conditional maintenance.

```python
engine.optimize_graph()
```

`optimize_graph()` checks whether the index needs rebuilding. If the update or delete ratio crosses the `delta_ratio` threshold, brinicle rebuilds the graph. Otherwise, it does nothing.

For unconditional compaction, use `rebuild_compact()`.


---

## Loading an Existing Index

To load an existing index, create `VectorEngine` with the same `index_path`.

If the dimension is already stored in the index, use `dim=0`.

```python
engine = brinicle.VectorEngine(
    "vector_index",
    dim=0,
)

results = engine.search(Q, k=10)
```

Use this for read/search sessions after an index has already been built.

---

## HNSW Tuning Notes

`M`, `ef_construction`, and `ef_search` control the usual HNSW tradeoffs.

| Parameter         | Effect                                                            |
| ----------------- | ----------------------------------------------------------------- |
| `M`               | Higher values usually improve recall but increase index size       |
| `ef_construction` | Higher values usually improve graph quality but slow down build    |
| `ef_search`       | Higher values usually improve recall but slow down query latency   |
| `build_n_threads` | Higher values can speed up build but increase CPU usage            |
| `delta_ratio`     | Lower values trigger rebuild/optimization sooner after changes     |

For large indexes, tune `n_shards` and `n_jobs` together:

| Parameter  | Effect                                                            |
| ---------- | ----------------------------------------------------------------- |
| `n_shards` | Splits the index into multiple local shards during build           |
| `n_jobs`   | Searches multiple shards or batch queries in parallel              |

---

## Complete API Reference

### `init`

```python
engine.init(mode="build")
```

Starts a write session.

Supported modes:

```text
build
insert
upsert
```

---

### `ingest`

```python
engine.ingest(
    external_id,
    vec,
)
```

Adds one vector to the current write session.

---

### `finalize`

```python
engine.finalize(
    optimize=False,
    M=0,
    ef_construction=0,
    ef_search=0,
    build_n_threads=0,
    seed=0,
)
```

Completes the pending write session.

Passing `0` for build parameters means brinicle should use the parameters already configured on the engine.

---

### `search`

```python
engine.search(
    q,
    k=10,
    efs=64,
    threshold=float("inf"),
    n_jobs=1,
)
```

Returns external IDs.

---

### `search_with_distance`

```python
engine.search_with_distance(
    q,
    k=10,
    efs=64,
    threshold=float("inf"),
    n_jobs=1,
)
```

Returns `(external_id, distance)` pairs.

---

### `search_batch`

```python
engine.search_batch(
    Q,
    k=10,
    efs=64,
    threshold=float("inf"),
    n_jobs=1,
)
```

Runs batch search over multiple query vectors.

---

### `delete_items`

```python
engine.delete_items(
    external_ids,
    return_not_found=False,
)
```

Deletes vectors by external ID.

---

### `needs_rebuild`

```python
engine.needs_rebuild()
```

Returns whether the index has crossed its maintenance threshold.

---

### `rebuild_compact`

```python
engine.rebuild_compact(
    M=16,
    ef_construction=200,
    ef_search=64,
    build_n_threads=1,
    seed=0,
)
```

Rebuilds the index from alive records.

---

### `optimize_graph`

```python
engine.optimize_graph()
```

Runs conditional graph maintenance.

---

### `close`

```python
engine.close()
```

Closes loaded index resources.

---

### `destroy`

```python
engine.destroy()
```

Removes index files from disk.

---

## Properties

```python
engine.dim
```

Returns the vector dimension.

```python
engine.has_index
```

Returns whether the engine has a loaded index.
