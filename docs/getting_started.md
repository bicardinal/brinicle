# Getting Started

## What is brinicle?

brinicle is a C++ retrieval engine with a Python API, built around disk-first, low-RAM HNSW search.

It supports three search engines:

* `VectorEngine` for raw vector similarity search
* `ItemSearchEngine` for lexical, semantic, and hybrid item search
* `AutocompleteEngine` for autocomplete and query suggestions

brinicle is designed for large indexes and constrained environments where keeping the full index in RAM is not practical.

---

## Installation

Install from PyPI:

```bash
pip install brinicle
```

Or install from source:

```bash
git clone https://github.com/bicardinal/brinicle.git
cd brinicle
pip install -e .
```

brinicle currently targets Python 3.12.x.

---

## Quick Start

```python
import numpy as np
import brinicle

dim = 128

engine = brinicle.VectorEngine("my_index", dim=dim)

engine.init(mode="build")

for i in range(1000):
    vector = np.random.randn(dim).astype("float32")
    engine.ingest(str(i), vector)

engine.finalize()

query = np.random.randn(dim).astype("float32")

results = engine.search(query, k=10)

print(results)
```

`search(...)` returns a list of external IDs:

```python
["42", "19", "705"]
```

---

## Choosing an Engine

| Engine               | Use it for                                             |
| -------------------- | ------------------------------------------------------ |
| `VectorEngine`       | Raw vector similarity search                           |
| `ItemSearchEngine`   | Product, catalog, or structured item search            |
| `AutocompleteEngine` | Autocomplete, title suggestions, and query suggestions |

Use `VectorEngine` when you already have vectors.

Use `ItemSearchEngine` when your records have titles, categories, attributes, and optional semantic vectors.

Use `AutocompleteEngine` when you want prefix-like suggestions for queries, titles, or curated phrases.

---

## Shared Lifecycle

All brinicle engines follow the same lifecycle:

```python
engine.init(mode="build")
engine.ingest(...)
engine.finalize()

engine.search(...)
```

The supported write modes are:

| Mode     | Meaning                                                                         |
| -------- | ------------------------------------------------------------------------------- |
| `build`  | Build a new index                                                               |
| `insert` | Add new records to an existing index                                            |
| `upsert` | Replace records with the same external IDs, or insert them if they do not exist |

---

## Main Index and Delta Index

brinicle stores updates using a main index and a delta index.

The main index stores the primary HNSW graph. The delta index stores later inserts and upserts.

During search, brinicle searches both indexes, merges the results, filters deleted records, and returns the top matches.

This allows brinicle to support updates without rebuilding the full index after every insert.

---

## Search Results and Distances

Use `search(...)` to return external IDs only:

```python
results = engine.search(query, k=10)
```

Use `search_with_distance(...)` to return external IDs with distances:

```python
results = engine.search_with_distance(query, k=10)
```

Example:

```python
[("42", 0.183), ("19", 0.241)]
```

brinicle ranks results by ascending distance. Smaller distance means a better match.

`VectorEngine` supports these distance functions:

| Distance function      | Meaning                       |
| ---------------------- | ----------------------------- |
| `l2`                   | Squared Euclidean distance    |
| `cosine_distance`      | `1 - cosine_similarity(a, b)` |
| `dot_product_distance` | `-dot_product(a, b)`          |

---

## Batch Search

Batch search runs multiple queries and returns one result list per query.

```python
queries = np.random.randn(100, dim).astype("float32")

results = engine.search_batch(
    queries,
    k=10,
    n_jobs=4,
)
```

`n_jobs` controls parallel query execution when parallel execution is available.

---

## Updates: Insert, Upsert, and Delete

Insert new records:

```python
engine.init(mode="insert")

engine.ingest("new_id", vector)

engine.finalize()
```

Upsert records:

```python
engine.init(mode="upsert")

engine.ingest("existing_or_new_id", new_vector)

engine.finalize()
```

Delete records:

```python
engine.delete_items(["id1", "id2"])
```

Deletes are logical until the index is compacted.

---

## Rebuild, Compact, and Optimize

brinicle provides maintenance methods for updated indexes.

| Method              | Meaning                                                                           |
| ------------------- | --------------------------------------------------------------------------------- |
| `needs_rebuild()`   | Returns whether the index has enough update or delete drift to justify rebuilding |
| `rebuild_compact()` | Rebuilds the index from alive records and removes deleted records physically      |
| `optimize_graph()`  | Rebuilds only when the index crosses the configured maintenance threshold         |

`delta_ratio` controls when brinicle considers an index ready for maintenance.

---

## Common Configuration Parameters

| Parameter         | Meaning                                             |
| ----------------- | --------------------------------------------------- |
| `dim`             | Vector or encoded representation dimension          |
| `M`               | HNSW graph connectivity                             |
| `ef_construction` | Build-time search width                             |
| `ef_search`       | Query-time search width                             |
| `delta_ratio`     | Maintenance threshold for delta and deleted records |
| `build_n_threads` | Number of build threads                             |
| `seed`            | Random seed for graph construction                  |

Example:

```python
engine = brinicle.VectorEngine(
    "my_index",
    dim=384,
    M=48,
    ef_construction=1024,
    ef_search=512,
    delta_ratio=0.1,
)
```

---

## Item Search Example

`ItemSearchEngine` is used for structured records such as products or catalog items.

```python
import brinicle

engine = brinicle.ItemSearchEngine(
    "items_index",
    dim=96,
    alpha=0.0,
)

engine.init(mode="build")

engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Apple",
        "storage": "256GB",
    },
)

engine.ingest(
    external_id="p2",
    title="Samsung Galaxy S24 Ultra 512GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Samsung",
        "storage": "512GB",
    },
)

engine.finalize()

results = engine.search("iphone 15 pro", k=10)

print(results)
```

For lexical-only search, use `alpha=0.0`.

For semantic or hybrid search, provide `vector_dim` and pass vectors during ingest and search.

---

## Autocomplete Example

`AutocompleteEngine` is used for query, title, or suggestion autocomplete.

```python
import brinicle

ac = brinicle.AutocompleteEngine(
    "autocomplete_index",
    dim=48,
)

ac.init(mode="build")

ac.ingest("iphone 15 pro max", "iphone 15 pro max")
ac.ingest("iphone 15 case", "iphone 15 case")
ac.ingest("samsung s24 ultra", "samsung s24 ultra")

ac.finalize()

results = ac.search("iphone", k=5)

print(results)
```

---

## Closing and Destroying an Index

Close loaded index resources:

```python
engine.close()
```

Destroy the index files:

```python
engine.destroy()
```

`destroy()` removes the index from disk.
