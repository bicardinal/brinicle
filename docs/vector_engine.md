# Vector Engine

`VectorEngine` is the raw vector search engine in Brinicle.

Use it when you already have embeddings or numeric vectors and want approximate nearest neighbor search through a disk-first HNSW index.

`VectorEngine` supports:

* build
* insert
* upsert
* delete
* single-query search
* batch search
* search with distances
* compact rebuild
* graph optimization

---

## Constructor

```python
engine = brinicle.VectorEngine(
    index_path,
    dim,
    delta_ratio=0.10,
    M=16,
    ef_construction=200,
    ef_search=64,
    build_n_threads=1,
    seed=0,
    dist_func="l2",
)
```

### Parameters

| Parameter         | Meaning                                             |
| ----------------- | --------------------------------------------------- |
| `index_path`      | Base path for the index files                       |
| `dim`             | Vector dimension                                    |
| `delta_ratio`     | Maintenance threshold for delta and deleted records |
| `M`               | HNSW graph connectivity                             |
| `ef_construction` | Build-time search width                             |
| `ef_search`       | Default query-time search width                     |
| `build_n_threads` | Number of build threads                             |
| `seed`            | Random seed for graph construction                  |
| `dist_func`       | Distance function used by the index                 |

Example:

```python
engine = brinicle.VectorEngine(
    "vector_index",
    dim=384,
    M=48,
    ef_construction=1024,
    ef_search=512,
    delta_ratio=0.1,
)
```

---

## Distance Functions

`VectorEngine` supports these distance functions:

| `dist_func`              | Meaning                       |
| ------------------------ | ----------------------------- |
| `"l2"`                   | Squared Euclidean distance    |
| `"cosine_distance"`      | `1 - cosine_similarity(a, b)` |
| `"dot_product_distance"` | `-dot_product(a, b)`          |

Brinicle ranks results by ascending distance.

Smaller distance means a better match.

For `dot_product_distance`, a larger dot product becomes a smaller distance:

```text
dot_product = 0.90  -> distance = -0.90
dot_product = 0.20  -> distance = -0.20
```

So the result with distance `-0.90` is ranked before the result with distance `-0.20`.

---

## Building an Index

Use `build` mode to create a new index.

```python
import numpy as np
import brinicle

dim = 128

engine = brinicle.VectorEngine("vector_index", dim=dim)

engine.init(mode="build")

for i in range(1000):
    vector = np.random.randn(dim).astype("float32")
    engine.ingest(str(i), vector)

engine.finalize()
```

Vectors must be one-dimensional `float32` arrays with the same dimension as the index.

---

## Search

Use `search(...)` to return external IDs only.

```python
query = np.random.randn(dim).astype("float32")

results = engine.search(query, k=10)

print(results)
```

Example output:

```python
["42", "18", "901"]
```

### Search Parameters

```python
engine.search(
    q,
    k=10,
    efs=64,
    threshold=float("inf"),
)
```

| Parameter   | Meaning                   |
| ----------- | ------------------------- |
| `q`         | Query vector              |
| `k`         | Maximum number of results |
| `efs`       | Query-time search width   |
| `threshold` | Maximum accepted distance |

Increasing `efs` usually improves recall, but increases query latency.

---

## Search with Distance

Use `search_with_distance(...)` to return both IDs and distances.

```python
results = engine.search_with_distance(query, k=10)

print(results)
```

Example output:

```python
[("42", 0.183), ("18", 0.241)]
```

The result format is:

```python
[(external_id, distance), ...]
```

---

## Batch Search

Use `search_batch(...)` to search multiple query vectors.

```python
queries = np.random.randn(100, dim).astype("float32")

results = engine.search_batch(
    queries,
    k=10,
    efs=64,
    n_jobs=4,
)
```

`queries` must be a two-dimensional `float32` array:

```text
(num_queries, dim)
```

The return value contains one result list per query:

```python
[
    ["42", "18", "901"],
    ["7", "103", "88"],
    ...
]
```

`n_jobs` controls parallel query execution when parallel execution is available.

---

## Insert

Use `insert` mode to add new vectors to an existing index.

```python
engine.init(mode="insert")

engine.ingest("new_id", new_vector)

engine.finalize()
```

Inserted records are added through the delta index. This allows Brinicle to accept updates without rebuilding the full main index after every insert.

---

## Upsert

Use `upsert` mode to replace existing records or insert new records.

```python
engine.init(mode="upsert")

engine.ingest("id1", updated_vector)

engine.finalize()
```

If `"id1"` already exists, Brinicle marks the old record as deleted and inserts the new version.

If `"id1"` does not exist, it is inserted as a new record.

---

## Delete

Use `delete_items(...)` to delete records by external ID.

```python
deleted_count, not_found = engine.delete_items(
    ["id1", "id2"],
    return_not_found=True,
)

print(deleted_count)
print(not_found)
```

If `return_not_found=False`, the second returned value is `None`.

Deletes are logical until the index is compacted. Deleted records are filtered out during search, but their storage is reclaimed during compact rebuild.

---

## Finalize Options

`finalize(...)` completes a pending `build`, `insert`, or `upsert`.

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

Passing `0` for build parameters uses the engine defaults.

When `optimize=False`, inserts and upserts are absorbed into the delta index.

When `optimize=True`, Brinicle may rebuild the index if the projected delta size crosses the maintenance threshold controlled by `delta_ratio`.

---

## Rebuild and Compact

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

---

## Optimize Graph

Use `optimize_graph(...)` to run conditional maintenance.

```python
engine.optimize_graph()
```

`optimize_graph()` checks whether the index needs rebuilding. If the update or delete ratio crosses the `delta_ratio` threshold, Brinicle rebuilds the graph. Otherwise, it does nothing.

For unconditional compaction, use `rebuild_compact()`.

---

## Index State

```python
engine.has_index
```

Returns whether the engine currently has a loaded main or delta index.

```python
engine.dim
```

Returns the vector dimension of the index.

```python
engine.needs_rebuild()
```

Returns whether the index has enough delta or deleted records to justify a rebuild.

---

## Close and Destroy

Close loaded index resources:

```python
engine.close()
```

Destroy the index files:

```python
engine.destroy()
```

`destroy()` removes the index from disk.

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
engine.ingest(external_id, vector)
```

Adds one vector to the current pending write session.

Call `init(...)` before calling `ingest(...)`.

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

---

### `search`

```python
engine.search(
    q,
    k=10,
    efs=64,
    threshold=float("inf"),
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

Runs batch search over a two-dimensional query matrix.

---

### `delete_items`

```python
engine.delete_items(
    external_ids,
    return_not_found=False,
)
```

Deletes records by external ID.

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
