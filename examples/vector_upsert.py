import numpy as np
import brinicle

dim = 4

engine = brinicle.VectorEngine("examples/indexes/vector_upsert", dim=dim)

engine.init("build")
engine.ingest("item-1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
engine.ingest("item-2", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
engine.finalize()

query = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

print("before upsert:", engine.search(query, k=2))

engine.init("upsert")
engine.ingest("item-1", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
engine.finalize()

print("after upsert:", engine.search(query, k=2))