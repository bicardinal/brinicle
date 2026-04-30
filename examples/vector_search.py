import numpy as np
import brinicle

dim = 4
vectors = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
], dtype=np.float32)

query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

engine = brinicle.VectorEngine("examples/indexes/vector_search", dim=dim)

engine.init("build")
for i, vec in enumerate(vectors):
    engine.ingest(str(i), vec)
engine.finalize()

print(engine.search(query, k=2))
print(engine.search_with_distance(query, k=2))