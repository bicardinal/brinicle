![Version 0.0.5](https://img.shields.io/badge/Version-0.0.5-green.svg)
![Python 3.12.x](https://img.shields.io/badge/Python-3.12.x-green.svg)
![Apache-2.0 License](https://img.shields.io/badge/License-Apache2.0-green.svg)


# brinicle

**brinicle** is a C++ retrieval engine built around **disk-first, low-RAM HNSW search**.

It supports:

- raw vector similarity search
- structured item search
- autocomplete/query suggestion search

## Benchmark

brinicle is designed for constrained environments where loading the full index into RAM is not practical.

In a 256MB RAM / 1 CPU container on MNIST 60K vectors, the benchmark result was:

| System | Outcome |
|---|---|
| brinicle | PASS |
| chroma | PASS |
| qdrant | OOMKilled |
| weaviate | OOMKilled |
| milvus | OOMKilled |

On SIFT 1M vectors, using the same in-process deployment model as FAISS and hnswlib:

| System | Build (s) | Recall@10 | Avg latency (ms) | QPS |
|---|---:|---:|---:|---:|
| faiss | 237.282 | 0.96999 | 0.092 | 10857.43 |
| hnswlib | 241.301 | 0.96364 | 0.093 | 10711.86 |
| brinicle | 243.75 | 0.96989 | 0.103 | 9730.65 |

In brinicle's benchmark suite, it reaches latency competitive with FAISS and hnswlib while keeping the index disk-backed and memory usage low.

![Memory usage comparison](https://brinicle.bicardinal.com/blow/memory_bars.png)

See the benchmark: [brinicle benchmark](https://brinicle.bicardinal.com/benchmark)

---

brinicle is designed for constrained environments where loading a full index into RAM is not practical. It keeps the same simple lifecycle across all engines:

```python
client.init(...)
client.ingest(...)
client.finalize()
client.search(...)
````

## Features

* Disk-first HNSW vector search
* Low-RAM indexing and querying
* Streaming-first ingest: one vector/item/suggestion at a time
* Insert, upsert, delete, and compact rebuild
* Raw vector search through `VectorEngine`
* Structured item search through `ItemSearchEngine`
* Autocomplete/query suggestion search through `AutocompleteEngine`
* Custom scoring for lexical item search and autocomplete
* Python bindings over a C++ core

---

## Install

Install from PyPI:

```bash
pip install brinicle
```

Or build from source:

```bash
git clone https://github.com/bicardinal/brinicle.git
cd brinicle
bash build.sh
```

---

## Engines

brinicle exposes three engines with the same lifecycle.

| Engine               | Use case                       | Input                                    |
| -------------------- | ------------------------------ | ---------------------------------------- |
| `VectorEngine`       | Raw ANN vector search          | `float32` vectors                        |
| `ItemSearchEngine`   | Structured catalog/item search | title, category, subcategory, attributes |
| `AutocompleteEngine` | Query/title suggestions        | suggestion text                          |

All engines follow the same pattern:

```python
client.init(mode="build")

for record in records:
    client.ingest(...)

client.finalize()

results = client.search(...)
```

---

## Vector search

Use `VectorEngine` when you already have embeddings or numeric vectors.

```python
import numpy as np
import brinicle

D = 2
n = 5

X = np.random.randn(n, D).astype(np.float32)
Q = np.random.randn(D).astype(np.float32)

engine = brinicle.VectorEngine(
    "vector_index",
    dim=D,
    delta_ratio=0.1,
)

engine.init(mode="build")

for eid in range(n):
    engine.ingest(str(eid), X[eid])

engine.finalize()

print(engine.search(Q, k=10))
```

`search(...)` returns a list of external ids:

```python
["3", "1", "0"]
```

To return distances too:

```python
print(engine.search_with_distance(Q, k=10))
```

---

## Insert

```python
Y = np.random.randn(5, D).astype(np.float32)

engine.init(mode="insert")

for eid in range(5):
    engine.ingest(str(eid) + "x", Y[eid])

engine.finalize()

print(engine.search(Q, k=10))
```

---

## Upsert

```python
Y = np.random.randn(5, D).astype(np.float32)

engine.init(mode="upsert")

for eid in range(5):
    engine.ingest(str(eid), Y[eid])

engine.finalize()

print(engine.search(Q, k=10))
```

---

## Delete

```python
engine.delete_items(["1", "4"])

print(engine.search(Q, k=10))
```

---

## Rebuild / optimize

```python
engine.optimize_graph()

print(engine.search(Q, k=10))
```

---

## Item search

`ItemSearchEngine` searches structured catalog-like records without requiring a traditional inverted index.

Each item can contain:

* `title`
* `category`
* `subcategory`
* `attributes`

Only `title` is required. The other fields are optional.

Items are encoded internally into fixed-size numeric representations and searched through brinicle's HNSW graph using a structured lexical scorer.

```python
import brinicle

engine = brinicle.ItemSearchEngine(
    "item_index",
    dim=96,
)

engine.init(mode="build")

engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB Natural Titanium",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Apple",
        "storage": "256GB",
        "color": "Natural Titanium",
    },
)

engine.ingest(
    external_id="p2",
    title="Samsung Galaxy S24 Ultra 512GB Black",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Samsung",
        "storage": "512GB",
        "color": "Black",
    },
)

engine.finalize()

print(engine.search("iphone 15 pro max", k=10))
```

To return distances:

```python
print(engine.search_with_distance("iphone 15", k=10))
```

Example with structured query fields:

```python
results = engine.search(
    "iphone 15",
    category="Electronics",
    subcategory="Smartphones",
    attributes={
        "brand": "Apple",
    },
    k=10,
)
```

### What can Item Search be used for?

`ItemSearchEngine` is useful for structured catalog-like data such as:

* products
* movies
* books
* jobs
* real estate listings
* restaurants
* games
* records with titles and attributes

Item Search is not a neural embedding model. It uses structured symbolic encoding and a configurable scorer.

---

## Autocomplete

`AutocompleteEngine` provides low-RAM autocomplete and query suggestion search using brinicle's HNSW infrastructure.

It can be used to index:

* popular queries
* item titles
* category names
* curated suggestions

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

print(ac.search("iph", k=5))
```

`AutocompleteEngine` follows the same lifecycle as the other engines:

```python
ac.init(mode="build")
ac.ingest(...)
ac.finalize()
ac.search(...)
```

The current autocomplete implementation is experimental and works best when query prefixes align well with encoded token prefixes.

---

## Streaming-first ingest

brinicle does not require loading the full dataset into memory.

Ingest is intentionally one record at a time:

```python
client.init(mode="build")

for item in stream_items():
    client.ingest(...)

client.finalize()
```

Users can stream data from:

* JSONL files
* databases
* APIs
* object storage
* custom pipelines

brinicle does not assume that your dataset fits in RAM. Rare in modern software.

---

## Configuration

brinicle exposes common HNSW parameters:

* `M`
* `ef_construction`
* `ef_search`
* `delta_ratio`

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

Item search also supports lexical scoring configuration.

```python
cfg = brinicle.LexicalConfig()

cfg.search_title_weight = 0.60
cfg.search_category_weight = 0.15
cfg.search_subcategory_weight = 0.15

engine = brinicle.ItemSearchEngine(
    "item_index",
    dim=96,
    lexical_config=cfg,
)
```

Autocomplete also supports its own scoring configuration.

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

---

## Index files

For an index path such as:

```python
engine = brinicle.VectorEngine("my_index", dim=128)
```

brinicle stores index files beside that base path:

```text
my_index.main
my_index.delta
my_index.lock
```

High-level engines such as `ItemSearchEngine` and `AutocompleteEngine` may also store metadata such as tokenizer and encoding information beside the index.

---

## Which engine should I use?

Use `VectorEngine` if you already have embeddings or numeric vectors.

Use `ItemSearchEngine` if you have structured catalog-like data such as products, movies, books, jobs, listings, or records with titles and attributes.

Use `AutocompleteEngine` if you want low-RAM query or title suggestions.

---

## Limitations

* brinicle is not a full-text search engine.
* Item Search is designed for structured catalog-like records, not long documents.
* Item Search is symbolic/lexical, not neural semantic search.
* Autocomplete is experimental.
* Search quality depends on normalization, tokenizer behavior, and field structure.
* Large updates may require graph optimization or compact rebuild.

---

## Roadmap

* High-level item search API
* High-level autocomplete API
* Metadata persistence for tokenizer and encoding config
* More benchmarks for item search and autocomplete
* Better prefix-aware autocomplete encoding
* Improved documentation and examples

---

## License

brinicle is licensed under the Apache License, Version 2.0.

See the [LICENSE](https://github.com/bicardinal/brinicle/blob/main/LICENSE) file.
