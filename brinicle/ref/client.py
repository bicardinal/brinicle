from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import orjson
import requests


class VectorEngineClient:
    def __init__(self, base_url: str = "http://localhost:1984"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def create_index(
        self,
        index_name: str,
        dim: int,
        delta_ratio: float = 0.10,
        ef_construction: int = 200,
        ef_search: int = 64,
        M: int = 16,
        seed: int = 0,
    ) -> Dict[str, Any]:
        payload = {"index_name": index_name, "dim": dim, "delta_ratio": delta_ratio}
        payload["params"] = {
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "M": M,
            "rng_seed": seed,
        }

        response = self.session.post(f"{self.base_url}/indexes", json=payload)
        response.raise_for_status()
        return response.json()

    def list_indexes(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/indexes")
        response.raise_for_status()
        return response.json()

    def delete_index(self, index_name: str, destroy: bool = False) -> Dict[str, Any]:
        response = self.session.delete(
            f"{self.base_url}/indexes/{index_name}", params={"destroy": destroy}
        )
        response.raise_for_status()
        return response.json()

    def load_index(self, index_name: str) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/indexes/load", json={"index_name": index_name}
        )
        response.raise_for_status()
        return response.json()

    def get_status(self, index_name: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/indexes/{index_name}/status")
        response.raise_for_status()
        return response.json()

    def init(self, index_name: str, mode: str) -> Dict[str, Any]:
        payload = {"index_name": index_name, "mode": mode}
        response = self.session.post(f"{self.base_url}/init", json=payload)
        response.raise_for_status()
        return response.json()

    def ingest(
        self, index_name: str, external_id: str, vector: np.ndarray
    ) -> Dict[str, Any]:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        payload = orjson.dumps(
            {"index_name": index_name, "external_id": external_id, "vector": vector}
        )

        response = self.session.post(
            f"{self.base_url}/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def ingest_batch_binary(
        self,
        index_name: str,
        ids: List,
        vectors: np.ndarray,
    ) -> Dict[str, Any]:
        payload = b""
        for vid, vec in enumerate(vectors):
            vid = ids[vid]
            id_bytes = vid.encode("ascii")[:8].ljust(8, b"\x00")
            payload += id_bytes

            if not isinstance(vec, np.ndarray):
                vec = np.array(vec, dtype=np.float32)
            elif vec.dtype != np.float32:
                vec = vec.astype(np.float32)

            payload += vec.tobytes()

        response = self.session.post(
            f"{self.base_url}/ingest/batch",
            params={"index_name": index_name},
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        return orjson.loads(response.content)

    def finalize(
        self,
        index_name: str,
        build_params: Optional[Dict[str, Any]] = None,
        optimize: bool = False,
    ) -> Dict[str, Any]:
        payload = {"index_name": index_name, "optimize": optimize}
        if build_params:
            payload["build_params"] = build_params

        response = self.session.post(f"{self.base_url}/finalize", json=payload)
        response.raise_for_status()
        return response.json()

    def delete_items(
        self, index_name: str, external_ids: List[str], return_not_found: bool = False
    ) -> Dict[str, Any]:
        payload = {
            "index_name": index_name,
            "external_ids": external_ids,
            "return_not_found": return_not_found,
        }
        response = self.session.post(f"{self.base_url}/delete", json=payload)
        response.raise_for_status()
        return response.json()

    def rebuild(
        self, index_name: str, build_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = {"index_name": index_name}
        if build_params:
            payload["build_params"] = build_params

        response = self.session.post(f"{self.base_url}/rebuild", json=payload)
        response.raise_for_status()
        return response.json()

    def search(self, index_name: str, query: np.ndarray, k: int = 10) -> List[str]:
        r = self.session.post(
            f"{self.base_url}/search.bin",
            params={"index_name": index_name, "k": k},
            data=query.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        neighbors = r.json()
        return neighbors

    def optimize(self, index_name: str) -> Dict[str, Any]:
        payload = orjson.dumps({"index_name": index_name})
        response = self.session.post(
            f"{self.base_url}/optimize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with VectorEngineClient() as client:
        client.create_index("test_index", dim=128)

        client.init("test_index", "build")

        vec = np.random.randn(128).astype(np.float32)
        client.ingest("test_index", "vec001", vec)

        vectors = [
            ("vec002", np.random.randn(128).astype(np.float32)),
            ("vec003", np.random.randn(128).astype(np.float32)),
        ]
        client.ingest_batch_binary("test_index", vectors)

        client.finalize("test_index", optimize=True)

        query = np.random.randn(128).astype(np.float32)
        results = client.search("test_index", query, k=5)
        print(f"Search results: {results}")

        status = client.get_status("test_index")
        print(f"Status: {status}")
