# brinicle
**brinicle** is a **C++ vector index engine (ANN library)** optimized for **disk-first, low-RAM** similarity search. It provides fast build + query, supports inserts/upserts/deletes, and targets predictable latency at high recall with minimal memory overhead on constrained environments.


## Install

You can either download via pip:
```bash
pip install brinicle
```
Or compile the code:
```bash
git clone https://github.com/bicardinal/brinicle.git
cd brinicle
bash build.sh
```

## Usage

```python
import numpy as np
import brinicle

D = 2
n = 5
X = np.random.randn(n, D).astype(np.float32)
Q = np.random.randn(D).astype(np.float32)

engine = brinicle.VectorEngine("test_index", dim=D, delta_ratio=0.1)

engine.init(mode="build")
for eid in range(n):
    engine.ingest(str(eid), X[eid])
engine.finalize()

print(engine.search(Q, k=10)) # returns a list of ids
```

To insert:
```python
Y = np.random.randn(5, D).astype(np.float32)
engine.init(mode="insert")
for eid in range(5):
    engine.ingest(str(eid) + "x", Y[eid])
engine.finalize()
print(engine.search(Q, k=10))
```

To upsert:
```python
Y = np.random.randn(5, D).astype(np.float32)
engine.init(mode="upsert")
for eid in range(5):
    engine.ingest(str(eid), Y[eid])
engine.finalize()
print(engine.search(Q, k=10))
```

To delete:
```python
engine.delete_items(["1", "4"])
print(engine.search(Q, k=10))
```

To re-build:
```python
engine.optimize()
print(engine.search(Q, k=10))
```
