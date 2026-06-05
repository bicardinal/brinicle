# Item Search Engine

`ItemSearchEngine` is brinicle's high-level engine for structured item search.

Use it when records have titles, categories, subcategories, attributes, and optionally semantic vectors.

It supports:

* lexical item search
* semantic item search
* hybrid lexical + semantic search
* single-query search
* batch search
* search with distances
* local sharding for large indexes
* insert, upsert, and delete
* compact rebuild and graph optimization

`ItemSearchEngine` uses the same disk-first HNSW infrastructure as `VectorEngine`, but it encodes structured item fields internally before indexing and searching.

---

## Constructor

```python
engine = brinicle.ItemSearchEngine(
    index_path,
    dim=96,
    vector_dim=0,
    vector_normalized=False,
    tokenizer_path=None,
    text_prep=None,
    title_ratio=0.9,
    delta_ratio=0.10,
    M=16,
    ef_construction=200,
    ef_search=64,
    build_n_threads=1,
    alpha=0.95,
    seed=0,
    lexical_config=None,
    n_shards=1
)
```

### Parameters

| Parameter           | Meaning                                                     |
| ------------------- | ----------------------------------------------------------- |
| `index_path`        | Base path for the index files                               |
| `dim`               | Dimension used for the encoded lexical representation       |
| `vector_dim`        | Dimension of optional semantic vectors                      |
| `vector_normalized` | Whether semantic vectors are already normalized             |
| `tokenizer_path`    | Optional custom tokenizer path                              |
| `text_prep`         | Optional text preprocessing                                 |
| `title_ratio`       | Portion of lexical encoding space reserved for title tokens |
| `delta_ratio`       | Maintenance threshold for delta and deleted records         |
| `M`                 | HNSW graph connectivity                                     |
| `ef_construction`   | Build-time search width                                     |
| `ef_search`         | Default query-time search width                             |
| `build_n_threads`   | Number of build threads                                     |
| `alpha`             | Balance between lexical and semantic scoring                |
| `seed`              | Random seed for graph construction                          |
| `lexical_config`    | Optional custom lexical scoring configuration               |
| `n_shards`          | Number of local shards used when building the index         |


Example:

```python
engine = brinicle.ItemSearchEngine(
    "items_index",
    dim=96,
    vector_dim=384,
    alpha=0.95,
    M=48,
    ef_construction=1024,
    ef_search=512,
)
```

---

## Local Sharding

`n_shards` controls how many local shards brinicle creates for the index during build.

The current sharding implementation is local sharding. It is not distributed sharding across multiple machines.

During insertion, brinicle uses hash-based routing to decide which shard receives each item. During search, brinicle searches all shards, then merges the partial results into a final ranked result list.

For multi-shard indexes, search methods accept `n_jobs`, which controls how many shards are searched in parallel.

Higher `n_jobs` can improve search speed, but it also increases CPU and I/O usage.

Example:

```python
engine = brinicle.ItemSearchEngine(
    "large_item_index",
    dim=96,
    vector_dim=384,
    alpha=1.0,
    n_shards=50,
)

results = engine.search(
    "",
    vector=query_vector,
    k=10,
    n_jobs=8,
)
```

As a practical starting point, keep `n_shards=1` for smaller indexes. Sharding becomes more useful when the index contains more than about 2 million elements (that's a rule of thumb, of course). For example, for an index with around 100 million elements, `n_shards=50` can be a reasonable starting point.

Benchmark `n_shards` and `n_jobs` with your real data and hardware, because the best values depend on item count, vector dimension, storage speed, and available CPU cores.

---

## Item Format

Each item has a required `title` and optional structured fields.

```python
engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Apple",
        "storage": "256GB",
        "color": "Natural Titanium",
    },
)
```

Only `title` is required. For pure vector-only indexes, this can be an empty string.

`category`, `subcategory`, `attributes`, and `vector` are optional.

For semantic or hybrid search, pass a vector during ingest:

```python
engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Apple"},
    vector=item_vector,
    normalize=True,
)
```

---

## Search Modes

`ItemSearchEngine` can be used in three common modes.

| Mode                                  | Setup                                                               |
| ------------------------------------- | ------------------------------------------------------------------- |
| Lexical search                        | Use structured fields and set `alpha=0.0`                           |
| Semantic search                       | Provide vectors and set `alpha=1.0`                                 |
| Hybrid search                         | Provide structured fields and vectors, then use `0.0 < alpha < 1.0` |

---

## Understanding `alpha`

`alpha` controls the balance between semantic vector similarity and lexical matching.

| `alpha` | Behavior                                 |
| ------: | ---------------------------------------- |
|   `0.0` | Lexical-only                             |
|   `0.5` | Balanced lexical + semantic              |
|  `0.95` | Mostly semantic, with lexical correction |
|   `1.0` | Semantic-only                            |

`alpha` affects both graph construction and search scoring, so choose it before building the index.

When `lexical_config` is provided, the custom config controls the weights directly.

 - If you only need vector search, with no title query and no parameter filtering, use `alpha=1.0` and pass an empty string as the query. This allows brinicle to use the Vector Fast Path during search, which can significantly improve search speed.
 - If you only need lexical search, with no vector and no parameter filtering, use `alpha=0.0`. This allows brinicle to use the Lexical Fast Path during search, which can significantly improve search speed.

---

## How Item Scoring Works

In general mode, brinicle combines several distance components:

```text
distance =
    title_weight       * title_distance
  + attribute_weight   * attribute_distance
  + category_weight    * category_distance
  + subcategory_weight * subcategory_distance
  + vector_weight      * vector_distance
```

Smaller distance means a better match.

| Component            | What it compares                           |
| -------------------- | ------------------------------------------ |
| Title distance       | Query/title token overlap                  |
| Attribute distance   | Attribute key-value matches and mismatches |
| Category distance    | Category match or mismatch                 |
| Subcategory distance | Subcategory match or mismatch              |
| Vector distance      | Semantic vector similarity                 |

---

## Title Matching

Title matching uses a Tversky-style distance over title tokens.

The main tuning parameters are:

| Parameter            | Effect                                                     |
| -------------------- | ---------------------------------------------------------- |
| `search_title_alpha` | Penalizes query tokens missing from the item title         |
| `search_title_beta`  | Penalizes extra item-title tokens not present in the query |
| `build_title_alpha`  | Same idea during graph construction                        |
| `build_title_beta`   | Same idea during graph construction                        |

Higher `alpha` makes missing query terms more expensive.

Higher `beta` makes extra item-title terms more expensive.

Lower `beta` is useful when short queries should match longer titles. For example, `"iphone 15"` can still match `"Apple iPhone 15 Pro Max 256GB"`.

---

## Attribute Matching

Attributes are key-value pairs.

```python
attributes={
    "brand": "Apple",
    "storage": "256GB",
}
```

At search time, if the query and item share an attribute key but the values differ, brinicle applies a very large distance penalty.

This allows attributes to behave like hard filters when `threshold` is used.

```python
results = engine.search(
    "iphone 15",
    attributes={"brand": "Apple"},
    threshold=10.0,
)
```

An item with `brand="Samsung"` receives a large penalty and can be filtered out by the threshold.

If an item does not have the queried attribute, brinicle applies a smaller penalty than a direct value mismatch.

---

## Category and Subcategory Matching

Category and subcategory mismatches use `category_penalty`.

If category or subcategory is missing on either side, no penalty is applied.

If both sides have values and they differ, brinicle applies the penalty.

The contribution is:

```text
search_category_weight * search_category_penalty
```

or:

```text
search_subcategory_weight * search_category_penalty
```

Example hard-filter style setup:

```python
cfg = brinicle.LexicalConfig()

cfg.search_category_weight = 1.0
cfg.search_category_penalty = 2.0

engine = brinicle.ItemSearchEngine(
    "items_index",
    dim=96,
    lexical_config=cfg,
)

results = engine.search(
    "iphone 15",
    category="Electronics",
    threshold=2.0,
)
```

brinicle accepts results with distance below the threshold. With this setup, a category mismatch contributes `2.0` by itself, so mismatched categories are filtered out.

The same approach can be used with subcategories.

---

## Vector Matching

Semantic vector matching is used when:

* `vector_dim > 0`
* vectors are provided during ingest
* a vector is provided during search
* vector weight is greater than zero

Example:

```python
engine = brinicle.ItemSearchEngine(
    "semantic_items",
    dim=96,
    vector_dim=384,
    alpha=1.0,
)
```

If your vectors are already normalized, use:

```python
engine = brinicle.ItemSearchEngine(
    "semantic_items",
    dim=96,
    vector_dim=384,
    alpha=1.0,
    vector_normalized=True,
)
```
Which can be faster.

For semantic and hybrid item search, query vectors must have the same dimension as `vector_dim`.

For pure vector search, pass an empty string as the item title during ingest and an empty string as the query during search. If there are no parameter filters, keep `alpha=1.0` so the engine can use the Vector Fast Path.

---

## Lexical Search Example

Lexical search does not require semantic vectors.

```python
import brinicle

engine = brinicle.ItemSearchEngine(
    "item_index",
    dim=96,
    alpha=0.0,
)

engine.init(mode="build")

engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Apple"},
)

engine.ingest(
    external_id="p2",
    title="Samsung Galaxy S24 Ultra 512GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Samsung"},
)

engine.finalize()

results = engine.search("iphone 15 pro", k=10)

print(results)
```

---

## Semantic Search Example

Semantic search uses vectors as the main retrieval signal.

```python
import numpy as np
import brinicle

vector_dim = 384

engine = brinicle.ItemSearchEngine(
    "semantic_item_index",
    dim=6,
    vector_dim=vector_dim,
    alpha=1.0,
)

engine.init(mode="build")

item_vector = np.random.randn(vector_dim).astype("float32")

engine.ingest(
    external_id="p1",
    title="",
    vector=item_vector,
    normalize=True,
)

engine.finalize()

query_vector = np.random.randn(vector_dim).astype("float32")

results = engine.search(
    "",
    vector=query_vector,
    normalize=True,
    k=10,
)

print(results)
```
`dim` is mainly used by lexical search and parameter filtering, if there is no parameter filtering, and no title, one can save memory by setting dim to 6, which is the minimum.
In this example, the empty search query tells brinicle not to use a title query. Because `alpha=1.0` and no category, subcategory, or attributes are provided, brinicle can use the Vector Fast Path.

---

## Vector Search with Parameter Filtering

Use this setup when retrieval should be vector-based, but category, subcategory, or attributes should still be used as filters.

In this mode, do not use the title as a lexical signal. Pass an empty string as the item title and query, then use a custom `LexicalConfig` that disables title weights at build time and search time.

```python
import numpy as np
import brinicle

vector_dim = 384

cfg = brinicle.LexicalConfig()
cfg.build_title_weight = 0.0
cfg.search_title_weight = 0.0

cfg.build_vector_weight = 1.0
cfg.search_vector_weight = 1.0

cfg.search_attr_weight = 0.2
cfg.search_category_weight = 0.3
cfg.search_subcategory_weight = 0.5

cfg.build_attr_weight = 0.2
cfg.build_category_weight = 0.3
cfg.build_subcategory_weight = 0.5

engine = brinicle.ItemSearchEngine(
    "vector_filter_item_index",
    dim=96,
    vector_dim=vector_dim,
    lexical_config=cfg,
)

engine.init(mode="build")

item_vector = np.random.randn(vector_dim).astype("float32")

engine.ingest(
    external_id="p1",
    title="",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Apple", "storage": "256GB"},
    vector=item_vector,
    normalize=True,
)

engine.finalize()

query_vector = np.random.randn(vector_dim).astype("float32")

results = engine.search(
    "",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Apple"},
    vector=query_vector,
    normalize=True,
    threshold=1.0,
    k=10,
)

print(results)
```

`threshold` controls how strict the filter behavior is. Tune the parameter weights, penalties, and threshold based on how hard the filters should be. `threshold=3.0` means complete hard-filtering.
Please note that for consistency, set weights of build, and search in such a way that they sums up to one. For instance, in this example 0.2 + 0.3 + 0.5 = 1.0 and the same thing for build.
In the background we have 1.0 score for semantic and 1.0 for lexical if weights sums up to 1.0. So, the worst match should have maximum distance 2.0.

---

## Hybrid Search Example

Hybrid search combines structured lexical fields with semantic vectors.

```python
import numpy as np
import brinicle

vector_dim = 384

engine = brinicle.ItemSearchEngine(
    "hybrid_item_index",
    dim=96,
    vector_dim=vector_dim,
    alpha=0.95,
)

engine.init(mode="build")

item_vector = np.random.randn(vector_dim).astype("float32")

engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Apple", "storage": "256GB"},
    vector=item_vector,
    normalize=True,
)

engine.finalize()

query_vector = np.random.randn(vector_dim).astype("float32")

results = engine.search(
    "iphone 15 pro",
    category="Electronics",
    attributes={"brand": "Apple"},
    vector=query_vector,
    normalize=True,
    k=10,
)

print(results)
```

---

## Search

```python
results = engine.search(
    query,
    k=10,
    efs=None,
    threshold=float("inf"),
    category=None,
    subcategory=None,
    attributes=None,
    vector=None,
    normalize=False,
    n_jobs=1,
)
```

### Parameters

| Parameter     | Meaning                               |
| ------------- | ------------------------------------- |
| `query`       | Text query                            |
| `k`           | Maximum number of results             |
| `efs`         | Query-time search width               |
| `threshold`   | Maximum accepted distance             |
| `category`    | Optional query category               |
| `subcategory` | Optional query subcategory            |
| `attributes`  | Optional query attributes             |
| `vector`      | Optional query vector                 |
| `normalize`   | Whether to normalize the query vector |
| `n_jobs`      | Number of shards to search in parallel on multi-shard indexes |

`search(...)` returns external IDs:

```python
["p1", "p7", "p3"]
```
`n_jobs` controls how many shards are searched in parallel. Higher values can reduce latency, but they also consume more CPU and I/O. For n_shards=1 setup, one can ignore this parameter.

---

## Search with Distance

Use `search_with_distance(...)` to return IDs and distances.

On multi-shard indexes, `n_jobs` controls how many shards are searched in parallel.

```python
results = engine.search_with_distance(
    "iphone 15 pro",
    category="Electronics",
    k=10,
)
```

Example output:

```python
[("p1", 0.12), ("p7", 0.19)]
```

---

## Batch Search

Use `search_batch(...)` to search multiple queries.

```python
results = engine.search_batch(
    queries,
    categories=categories,
    subcategories=subcategories,
    attributes_list=attributes_list,
    vectors=vectors,
    k=10,
    n_jobs=4,
)
```

If `categories`, `subcategories`, `attributes_list`, or `vectors` are provided, their length must match `len(queries)`.



Example:

```python
queries = [
    "iphone 15 pro",
    "running shoes",
]

categories = [
    "Electronics",
    "Fashion",
]

attributes_list = [
    {"brand": "Apple"},
    {"brand": "Nike"},
]

results = engine.search_batch(
    queries,
    categories=categories,
    attributes_list=attributes_list,
    k=10,
)
```

Batch search supports optimized paths for:

* query-only batch search
* query + vector batch search
* full per-query metadata batch search

---

## Insert

Use `insert` mode to add new items to an existing index.

```python
engine.init(mode="insert")

engine.ingest(
    external_id="p3",
    title="Google Pixel 8 Pro 256GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Google"},
)

engine.finalize()
```

Inserted items are added through the delta index.

---

## Upsert

Use `upsert` mode to replace existing items or insert new ones.

```python
engine.init(mode="upsert")

engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 512GB",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Apple",
        "storage": "512GB",
    },
)

engine.finalize()
```

If the external ID already exists, brinicle marks the old record as deleted and inserts the new version.

If the external ID does not exist, the item is inserted as a new record.

---

## Delete

Use `delete_items(...)` to delete items by external ID.

```python
deleted_count, not_found = engine.delete_items(
    ["p1", "p2"],
    return_not_found=True,
)

print(deleted_count)
print(not_found)
```

If `return_not_found=False`, the second returned value is `None`.

Deletes are logical until compact rebuild.

---

## Rebuild and Optimize

Item indexes use the same maintenance model as `VectorEngine`.

```python
engine.needs_rebuild()
```

Returns whether the index has enough update or delete drift to justify rebuilding.

```python
engine.rebuild_compact()
```

Rebuilds the index from alive records, removes deleted records physically, and clears the delta index.

```python
engine.optimize_graph()
```

Runs conditional maintenance. If the index crosses the configured maintenance threshold, brinicle rebuilds the graph.

---

## Custom LexicalConfig

Use `LexicalConfig` when you want direct control over item scoring.

```python
cfg = brinicle.LexicalConfig()

cfg.search_title_weight = 0.60
cfg.search_attr_weight = 0.10
cfg.search_category_weight = 0.15
cfg.search_subcategory_weight = 0.15
cfg.search_vector_weight = 0.10

engine = brinicle.ItemSearchEngine(
    "item_index",
    dim=96,
    lexical_config=cfg,
)
```

`LexicalConfig` has separate build-time and search-time weights.

Build-time weights affect graph construction.

Search-time weights affect query ranking.

---

## LexicalConfig Parameters

| Parameter                   | Effect                                                              |
| --------------------------- | ------------------------------------------------------------------- |
| `build_title_weight`        | Title weight during graph construction                              |
| `build_attr_weight`         | Attribute weight during graph construction                          |
| `build_category_weight`     | Category weight during graph construction                           |
| `build_subcategory_weight`  | Subcategory weight during graph construction                        |
| `build_vector_weight`       | Vector weight during graph construction                             |
| `search_title_weight`       | Title weight during search                                          |
| `search_attr_weight`        | Attribute weight during search                                      |
| `search_category_weight`    | Category weight during search                                       |
| `search_subcategory_weight` | Subcategory weight during search                                    |
| `search_vector_weight`      | Vector weight during search                                         |
| `build_category_penalty`    | Category and subcategory mismatch penalty during graph construction |
| `search_category_penalty`   | Category and subcategory mismatch penalty during search             |
| `build_title_alpha`         | Build-time title Tversky alpha                                      |
| `build_title_beta`          | Build-time title Tversky beta                                       |
| `search_title_alpha`        | Search-time title Tversky alpha                                     |
| `search_title_beta`         | Search-time title Tversky beta                                      |
| `vector_normalized`         | Whether vectors are already normalized                              |

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
engine.ingest(
    external_id,
    title,
    category=None,
    subcategory=None,
    attributes=None,
    vector=None,
    normalize=False,
)
```

Adds one item to the current write session.

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
    query,
    k=10,
    efs=None,
    threshold=float("inf"),
    category=None,
    subcategory=None,
    attributes=None,
    vector=None,
    normalize=False,
    n_jobs=1,
)
```

Returns external IDs.

---

### `search_with_distance`

```python
engine.search_with_distance(
    query,
    k=10,
    efs=None,
    threshold=float("inf"),
    category=None,
    subcategory=None,
    attributes=None,
    vector=None,
    normalize=False,
    n_jobs=1,
)
```

Returns `(external_id, distance)` pairs.

---

### `search_batch`

```python
engine.search_batch(
    queries,
    k=10,
    efs=None,
    threshold=float("inf"),
    categories=None,
    subcategories=None,
    attributes_list=None,
    vectors=None,
    normalize=False,
    n_jobs=1,
)
```

Runs batch search over multiple item queries.

---

### `delete_items`

```python
engine.delete_items(
    external_ids,
    return_not_found=False,
)
```

Deletes items by external ID.

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
