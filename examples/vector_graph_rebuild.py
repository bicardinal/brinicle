import numpy as np
import brinicle

dim = 4

engine = brinicle.VectorEngine("examples/indexes/vector_rebuild", dim=dim)

engine.init("build")
for i in range(dim):
    engine.ingest(str(i), np.eye(dim, dtype=np.float32)[i])
engine.finalize()

engine.delete_items(["1", "2"])

print("needs rebuild:", engine.needs_rebuild())

engine.optimize_graph()

query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

print("after rebuild:", engine.search(query, k=4))