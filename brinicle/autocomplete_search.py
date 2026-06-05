from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ._brinicle import AutocompleteConfig
from ._brinicle import VectorEngine
from .lexical_encoder import LexicalEncoder


class AutocompleteEngine:
    """
    High-level autocomplete/search-suggestion wrapper.

    This class keeps the same lifecycle as VectorEngine:

        init(...)
        ingest(...)
        finalize(...)
        search(...)
        search_with_distance(...)

    But unlike VectorEngine, it accepts text suggestions and text queries.
    Encoding is handled internally through LexicalEncoder.
    """

    def __init__(
        self,
        index_path: str | Path,
        dim: int = 48,
        *,
        tokenizer_path: str | Path | None = None,
        text_prep: Any = None,
        delta_ratio: float = 0.10,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        build_n_threads: int = 1,
        seed: int = 0,
        autocomplete_config: AutocompleteConfig | None = None,
        n_shards: int = 1,
    ) -> None:
        if dim <= 1:
            raise ValueError("dim must be greater than 1")

        self.build_n_threads = int(build_n_threads)

        self.index_path = str(index_path)
        self._dim = int(dim)
        self._ef_search = int(ef_search)
        self.n_shards = int(n_shards)
        self.encoder = LexicalEncoder(
            tokenizer_path=tokenizer_path,
            max_dim=self._dim,
            text_prep=text_prep,
        )

        self.autocomplete_config = (
            autocomplete_config
            if autocomplete_config is not None
            else AutocompleteConfig()
        )

        self._engine = VectorEngine(
            self.index_path,
            self._dim,
            delta_ratio,
            M,
            ef_construction,
            ef_search,
            build_n_threads=build_n_threads,
            seed=seed,
            dist_func="autocomplete",
            autocomplete_config=self.autocomplete_config,
            n_shards=self.n_shards,
        )

    def init(self, mode: str = "build") -> None:
        self._engine.init(mode)

    def ingest(
        self,
        external_id: str,
        text: str,
        *,
        normalize: bool = True,
    ) -> None:
        """
        Ingest one autocomplete suggestion.

        external_id is what search returns. It can be the suggestion text itself,
        a query id, an item id, or any caller-defined identifier.
        """

        vec = self.encoder.encode_build_autocomplete_vector(
            text,
            normalize=normalize,
        )

        self._engine.ingest(str(external_id), self._as_f32(vec))

    def finalize(
        self,
        optimize: bool = False,
        M: int = 0,
        ef_construction: int = 0,
        ef_search: int = 0,
        build_n_threads: int = 0,
        seed: int = 0,
    ) -> None:
        self._engine.finalize(
            optimize=optimize,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            build_n_threads=build_n_threads,
            seed=seed,
        )

    def search(
        self,
        query: str,
        k: int = 10,
        efs: int | None = None,
        threshold: float = math.inf,
        *,
        normalize: bool = True,
        n_jobs: int = 1,
    ) -> list[str]:
        qvec = self.encoder.encode_query_autocomplete_vector(
            query,
            normalize=normalize,
        )

        return self._engine.search(
            self._as_f32(qvec),
            k=k,
            efs=self._resolve_efs(efs),
            threshold=threshold,
            n_jobs=n_jobs,
        )

    def search_with_distance(
        self,
        query: str,
        k: int = 10,
        efs: int | None = None,
        threshold: float = math.inf,
        *,
        normalize: bool = True,
        n_jobs: int = 1,
    ) -> list[tuple[str, float]]:
        qvec = self.encoder.encode_query_autocomplete_vector(
            query,
            normalize=normalize,
        )

        return self._engine.search_with_distance(
            self._as_f32(qvec),
            k=k,
            efs=self._resolve_efs(efs),
            threshold=threshold,
            n_jobs=n_jobs,
        )

    def search_batch(
        self,
        queries: list[str],
        k: int = 10,
        efs: int | None = None,
        threshold: float = math.inf,
        *,
        normalize: bool = True,
        n_jobs: int = 1,
    ) -> list[list[str]]:
        """
        Batch autocomplete search for multiple independent queries.
        This method is experimental, and mostly for benchmarking.

        n_jobs:
            1  -> serial batch inside C++
            >1 -> parallel batch inside C++
            -1 -> OpenMP default, if enabled
        """

        if not queries:
            return []

        qmat = np.empty((len(queries), self._dim), dtype=np.float32)

        for i, query in enumerate(queries):
            qvec = self.encoder.encode_query_autocomplete_vector(
                str(query or ""),
                normalize=normalize,
            )
            qmat[i] = self._as_f32(qvec)

        return self._engine.search_batch(
            qmat,
            k=k,
            efs=self._resolve_efs(efs),
            threshold=threshold,
            n_jobs=n_jobs,
        )

    def delete_items(
        self,
        external_ids: list[str],
        return_not_found: bool = False,
    ):
        return self._engine.delete_items(
            [str(x) for x in external_ids],
            return_not_found=return_not_found,
        )

    def rebuild_compact(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        build_n_threads: int = 1,
        seed: int = 0,
    ) -> None:
        self._engine.rebuild_compact(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            build_n_threads=build_n_threads,
            seed=seed,
        )

    def needs_rebuild(self) -> bool:
        return self._engine.needs_rebuild()

    def optimize_graph(self) -> None:
        self._engine.optimize_graph()

    def close(self) -> None:
        self._engine.close()

    def destroy(self) -> None:
        self._engine.destroy()

    @property
    def dim(self) -> int:
        return self._engine.dim

    @property
    def has_index(self) -> bool:
        return self._engine.has_index

    def _as_f32(self, vec: Any) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)

        if arr.ndim != 1:
            raise ValueError("encoded vector must be 1-D")

        if arr.shape[0] != self._dim:
            raise ValueError(
                f"encoded vector dimension mismatch: expected {self._dim}, got {arr.shape[0]}"
            )

        return np.ascontiguousarray(arr, dtype=np.float32)

    def _resolve_efs(self, efs: int | None) -> int:
        if efs is None:
            return self._ef_search

        if efs <= 0:
            raise ValueError("efs must be greater than 0")

        return int(efs)

    def __enter__(self) -> "AutocompleteEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
