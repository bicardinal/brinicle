import numpy as np
import brinicle

dim = 4
vectors = np.eye(dim, dtype=np.float32)

query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

engine = brinicle.VectorEngine("examples/indexes/vector_delete", dim=dim)

engine.init("build")
for i, vec in enumerate(vectors):
    engine.ingest(str(i), vec)
engine.finalize()

print("before delete:", engine.search(query, k=4))

deleted, not_found = engine.delete_items(["1"], return_not_found=True)

print("deleted:", deleted)
print("not found:", not_found)
print("after delete:", engine.search(query, k=4))