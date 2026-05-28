Here is the clean draft for the **Autocomplete Engine** section.

# Autocomplete Engine

`AutocompleteEngine` is Brinicle’s high-level engine for autocomplete and suggestion search.

Use it for:

* query suggestions
* title suggestions
* product-name suggestions
* category suggestions
* curated search phrases

`AutocompleteEngine` uses the same disk-first HNSW infrastructure as the other Brinicle engines, but it encodes suggestion text internally before indexing and searching.

It supports:

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
ac = brinicle.AutocompleteEngine(
    index_path,
    dim=48,
    tokenizer_path=None,
    text_prep=None,
    delta_ratio=0.10,
    M=16,
    ef_construction=200,
    ef_search=64,
    build_n_threads=1,
    seed=0,
    autocomplete_config=None,
)
```

### Parameters

| Parameter             | Meaning                                             |
| --------------------- | --------------------------------------------------- |
| `index_path`          | Base path for the index files                       |
| `dim`                 | Dimension used for encoded autocomplete tokens      |
| `tokenizer_path`      | Optional custom tokenizer path                      |
| `text_prep`           | Optional text preprocessing                         |
| `delta_ratio`         | Maintenance threshold for delta and deleted records |
| `M`                   | HNSW graph connectivity                             |
| `ef_construction`     | Build-time search width                             |
| `ef_search`           | Default query-time search width                     |
| `build_n_threads`     | Number of build threads                             |
| `seed`                | Random seed for graph construction                  |
| `autocomplete_config` | Optional autocomplete scoring configuration         |

Example:

```python
ac = brinicle.AutocompleteEngine(
    "autocomplete_index",
    dim=48,
    M=32,
    ef_construction=512,
    ef_search=128,
)
```

---

## Suggestion Format

Each suggestion has an `external_id` and a `text` value.

```python
ac.ingest(
    "iphone 15 pro max",
    "iphone 15 pro max",
)
```

The `text` value is encoded and indexed.

The `external_id` is what search returns. It can be the suggestion text itself, an item ID, a query ID, or any caller-defined identifier.

Example with separate IDs:

```python
ac.ingest(
    "suggestion_001",
    "iphone 15 pro max",
)
```

---

## How Autocomplete Scoring Works

Autocomplete search is prefix-oriented.

Brinicle compares query tokens with suggestion tokens from the beginning of the text. Earlier token matches matter more than later token matches.

Example:

```text
query: "iphone 15"

good:  "iphone 15 pro max"
weaker: "case for iphone 15"
```

Both suggestions contain related terms, but the first one is prefix-aligned with the query.

Smaller distance means a better match.

---

## Query Length and Thresholds

`AutocompleteEngine` works cautiously. It usually gives clearer results when the query contains at least one complete term.

Very short partial inputs such as:

```python
"iph"
```

may not produce strong or obvious matches.

Queries such as:

```python
"iphone"
"iphone 15"
"samsung"
```

usually provide clearer signals.

Use `threshold` to filter weak or irrelevant suggestions:

```python
results = ac.search(
    "iphone",
    k=5,
    threshold=0.8,
)
```

Lower thresholds make autocomplete stricter. Higher thresholds allow more distant suggestions.

---

## AutocompleteConfig

Use `AutocompleteConfig` when you want direct control over autocomplete scoring.

```python
cfg = brinicle.AutocompleteConfig()

cfg.search_position_decay = 0.5
cfg.search_length_penalty = 0.2

ac = brinicle.AutocompleteEngine(
    "autocomplete_index",
    dim=48,
    autocomplete_config=cfg,
)
```

`AutocompleteConfig` has separate build-time and search-time parameters.

Build-time parameters affect graph construction.

Search-time parameters affect query ranking.

---

## AutocompleteConfig Parameters

| Parameter               | Effect                                                                   |
| ----------------------- | ------------------------------------------------------------------------ |
| `build_position_decay`  | Controls how much later token positions matter during graph construction |
| `search_position_decay` | Controls how much later token positions matter during search             |
| `build_length_penalty`  | Length penalty used during graph construction                            |
| `search_length_penalty` | Penalizes suggestions that are much longer than the query during search  |

`position_decay` controls how quickly later token positions lose importance.

Lower values make early tokens much more important.

Higher values make later tokens matter more.

`length_penalty` controls how strongly longer suggestions are penalized.

Higher values make autocomplete stricter toward long suggestions.

Lower values allow longer completions more easily.

---

## Building an Autocomplete Index

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
```

---

## Search

Use `search(...)` to return suggestion IDs.

```python
results = ac.search("iphone", k=5)

print(results)
```

Example output:

```python
["iphone 15 pro max", "iphone 15 case"]
```

### Search Parameters

```python
ac.search(
    query,
    k=10,
    efs=None,
    threshold=float("inf"),
    normalize=True,
)
```

| Parameter   | Meaning                                   |
| ----------- | ----------------------------------------- |
| `query`     | Text query                                |
| `k`         | Maximum number of suggestions             |
| `efs`       | Query-time search width                   |
| `threshold` | Maximum accepted distance                 |
| `normalize` | Whether to normalize encoded query values |

Increasing `efs` usually improves recall, but increases query latency.

---

## Search with Distance

Use `search_with_distance(...)` to return suggestion IDs and distances.

```python
results = ac.search_with_distance(
    "iphone",
    k=5,
)

print(results)
```

Example output:

```python
[("iphone 15 pro max", 0.0), ("iphone 15 case", 0.12)]
```

The result format is:

```python
[(external_id, distance), ...]
```

---

## Batch Search

Use `search_batch(...)` to search multiple autocomplete queries.

```python
results = ac.search_batch(
    ["iphone", "samsung", "laptop"],
    k=5,
    n_jobs=4,
)
```

The return value contains one suggestion list per query:

```python
[
    ["iphone 15 pro max", "iphone 15 case"],
    ["samsung s24 ultra"],
    ["laptop stand", "laptop sleeve"],
]
```

`n_jobs` controls parallel query execution when parallel execution is available.

---

## Insert

Use `insert` mode to add new suggestions to an existing index.

```python
ac.init(mode="insert")

ac.ingest("iphone 16 pro", "iphone 16 pro")
ac.ingest("iphone 16 pro case", "iphone 16 pro case")

ac.finalize()
```

Inserted suggestions are added through the delta index.

---

## Upsert

Use `upsert` mode to replace existing suggestions or insert new ones.

```python
ac.init(mode="upsert")

ac.ingest("iphone 15 pro max", "iphone 15 pro max 256gb")

ac.finalize()
```

If the external ID already exists, Brinicle marks the old record as deleted and inserts the new version.

If the external ID does not exist, the suggestion is inserted as a new record.

---

## Delete

Use `delete_items(...)` to delete suggestions by external ID.

```python
deleted_count, not_found = ac.delete_items(
    ["iphone 15 case", "missing"],
    return_not_found=True,
)

print(deleted_count)
print(not_found)
```

If `return_not_found=False`, the second returned value is `None`.

Deletes are logical until compact rebuild.

---

## Rebuild and Optimize

Autocomplete indexes use the same maintenance model as `VectorEngine`.

```python
ac.needs_rebuild()
```

Returns whether the index has enough update or delete drift to justify rebuilding.

```python
ac.rebuild_compact()
```

Rebuilds the index from alive records, removes deleted records physically, and clears the delta index.

```python
ac.optimize_graph()
```

Runs conditional maintenance. If the index crosses the configured maintenance threshold, Brinicle rebuilds the graph.

---

## Close and Destroy

Close loaded index resources:

```python
ac.close()
```

Destroy the index files:

```python
ac.destroy()
```

`destroy()` removes the index from disk.

---

## Complete API Reference

### `init`

```python
ac.init(mode="build")
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
ac.ingest(
    external_id,
    text,
    normalize=True,
)
```

Adds one suggestion to the current write session.

---

### `finalize`

```python
ac.finalize(
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
ac.search(
    query,
    k=10,
    efs=None,
    threshold=float("inf"),
    normalize=True,
)
```

Returns suggestion IDs.

---

### `search_with_distance`

```python
ac.search_with_distance(
    query,
    k=10,
    efs=None,
    threshold=float("inf"),
    normalize=True,
)
```

Returns `(external_id, distance)` pairs.

---

### `search_batch`

```python
ac.search_batch(
    queries,
    k=10,
    efs=None,
    threshold=float("inf"),
    normalize=True,
    n_jobs=1,
)
```

Runs batch autocomplete search.

---

### `delete_items`

```python
ac.delete_items(
    external_ids,
    return_not_found=False,
)
```

Deletes suggestions by external ID.

---

### `needs_rebuild`

```python
ac.needs_rebuild()
```

Returns whether the index has crossed its maintenance threshold.

---

### `rebuild_compact`

```python
ac.rebuild_compact(
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
ac.optimize_graph()
```

Runs conditional graph maintenance.

---

### `close`

```python
ac.close()
```

Closes loaded index resources.

---

### `destroy`

```python
ac.destroy()
```

Removes index files from disk.
