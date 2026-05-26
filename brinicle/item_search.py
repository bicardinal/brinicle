from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Union
from typing import Tuple

import numpy as np

from ._brinicle import LexicalConfig
from ._brinicle import VectorEngine
from .lexical_encoder import LexicalEncoder

Label = Optional[Union[str, int]]
Attributes = Optional[Mapping[str, Any]]


class ItemSearchEngine:
    """
    High-level structured item search wrapper.

    This class keeps the same lifecycle as VectorEngine:

        init(...)
        ingest(...)
        finalize(...)
        search(...)
        search_with_distance(...)

    But unlike VectorEngine, it accepts structured items and queries.
    Encoding is handled internally through LexicalEncoder.
    """

    def __init__(
        self,
        index_path: str | Path,
        dim: int = 96,
        vector_dim: int = 0,
        vector_normalized: bool = False,
        *,
        tokenizer_path: str | Path | None = None,
        text_prep: Any = None,
        title_ratio: float = 0.9,
        delta_ratio: float = 0.10,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        build_n_threads: int = 1,
        alpha: float = 0.95,
        seed: int = 0,
        lexical_config: LexicalConfig | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be greater than 0")

        self.index_path = str(index_path)
        self._dim = int(dim)
        self._ef_search = int(ef_search)
        self.vector_dim = int(vector_dim)

        self.encoder = LexicalEncoder(
            tokenizer_path=tokenizer_path,
            max_dim=self._dim,
            vector_dim=self.vector_dim,
            text_prep=text_prep,
            title_ratio=title_ratio,
        )
        self.alpha = alpha
        self._adjust_weights(lexical_config, alpha)

        self.lexical_config.vector_normalized = vector_normalized

        self._engine = VectorEngine(
            self.index_path,
            self._dim + self.vector_dim,
            delta_ratio,
            M,
            ef_construction,
            ef_search,
            build_n_threads=build_n_threads,
            seed=seed,
            dist_func="lexical",
            lexical_config=self.lexical_config,
        )

    def _alpha_weight(self, p: float) -> Tuple[float, float]:
        """
        Converts a semantic-bias parameter into absolute vector weight
        and relative lexical correction weight.

        The semantic/vector side is treated as the smooth primary geometry.
        The lexical side is a correction term applied on top of each lexical
        field's default base weight.

        p=1.00 -> vector=1.0, lexical=0.0
        p=0.95 -> vector=1.0, lexical≈0.0526 before field base weights
        p=0.50 -> vector=1.0, lexical=1.0 before field base weights
        p=0.00 -> vector=0.0, lexical=1.0
        """

        p = float(p)
        if p <= 0.0:
            return 0.0, 1.0
        if p >= 1.0:
            return 1.0, 0.0
        return 1.0, (1.0 - p) / p

    def _adjust_weights(self, lexical_config, alpha):
        if lexical_config is None:
            semantic_weight, lexical_weight = self._alpha_weight(alpha)
            c = LexicalConfig()
            c.search_attr_weight *= lexical_weight
            c.search_category_weight *= lexical_weight
            c.search_subcategory_weight *= lexical_weight
            c.search_title_weight *= lexical_weight
            c.search_vector_weight = semantic_weight

            c.build_attr_weight *= lexical_weight
            c.build_category_weight *= lexical_weight
            c.build_subcategory_weight *= lexical_weight
            c.build_title_weight *= lexical_weight
            c.build_vector_weight = semantic_weight
            self.lexical_config = c
        else:
            self.lexical_config = lexical_config

    def init(self, mode: str = "build") -> None:
        self._engine.init(mode)

    def ingest(
        self,
        external_id: str,
        title: str,
        category: Label = None,
        subcategory: Label = None,
        attributes: Attributes = None,
        vector: np.ndarray = None,
        *,
        normalize: bool = False,
    ) -> None:
        """
        Ingest one structured item.
        """

        vec = self._encode_item(
            title=title,
            attributes=attributes,
            category=category,
            subcategory=subcategory,
            vector=vector,
            normalize=normalize,
        )
        self._engine.ingest(str(external_id), vec)

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
        category: Label = None,
        subcategory: Label = None,
        attributes: Attributes = None,
        vector: np.ndarray = None,
        *,
        normalize: bool = False,
    ) -> list[str]:

        qvec = self._encode_query(
            query=query,
            attributes=attributes,
            category=category,
            subcategory=subcategory,
            vector=vector,
            normalize=normalize,
        )

        return self._engine.search(
            qvec,
            k=k,
            efs=self._resolve_efs(efs),
            threshold=threshold,
        )

    def search_batch(
        self,
        queries: list[str],
        k: int = 10,
        efs: int | None = None,
        threshold: float = math.inf,
        category: Label = None,
        subcategory: Label = None,
        attributes: Attributes = None,
        vectors: np.ndarray = None,
        *,
        normalize: bool = False,
        n_jobs: int = 1,
    ) -> list[list[str]]:
        """
        Batch search for multiple independent queries.
        This method is experimental, and mostly for benchmarking.

        n_jobs:
            1  -> serial batch inside C++
            >1 -> parallel batch inside C++
            -1 -> OpenMP default, if enabled
        """

        if not queries:
            return []

        qmat = np.empty((len(queries), self._dim + self.vector_dim), dtype=np.float32)

        for i, query in enumerate(queries):
            qvec = self._encode_query(
                query=str(query or ""),
                attributes=attributes,
                category=category,
                subcategory=subcategory,
                vector=vectors[i],
                normalize=normalize,
            )
            qmat[i] = qvec

        return self._engine.search_batch(
            qmat,
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
        category: Label = None,
        subcategory: Label = None,
        attributes: Attributes = None,
        vector: np.ndarray = None,
        *,
        normalize: bool = False,
    ) -> list[tuple[str, float]]:

        qvec = self._encode_query(
            query=query,
            attributes=attributes,
            category=category,
            subcategory=subcategory,
            vector=vector,
            normalize=normalize,
        )

        return self._engine.search_with_distance(
            qvec,
            k=k,
            efs=self._resolve_efs(efs),
            threshold=threshold,
        )

    def delete_items(
        self,
        external_ids: list[str],
        return_not_found: bool = False,
    ):
        return self._engine.delete_items(
            external_ids, return_not_found=return_not_found
        )

    def rebuild_compact(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 0,
    ) -> None:
        self._engine.rebuild_compact(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
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

    def _encode_item(
        self,
        *,
        title: str,
        attributes: Attributes,
        category: Label,
        subcategory: Label,
        vector: np.ndarray,
        normalize: bool,
    ) -> np.ndarray:
        attrs = self._normalize_attributes(attributes)
        return self.encoder.encode_item_vector(
            title=title,
            attributes=attrs,
            category=category,
            subcategory=subcategory,
            vector=vector,
            normalize=normalize,
        )

    def _encode_query(
        self,
        *,
        query: str,
        attributes: Attributes,
        category: Label,
        subcategory: Label,
        vector: np.ndarray,
        normalize: bool,
    ) -> np.ndarray:
        attrs = self._normalize_attributes(attributes)
        return self.encoder.encode_query_vector(
            query=query,
            attributes=attrs,
            category=category,
            subcategory=subcategory,
            vector=vector,
            normalize=normalize,
        )

    def _resolve_efs(self, efs: int | None) -> int:
        if efs is None:
            return self._ef_search

        if efs <= 0:
            raise ValueError("efs must be greater than 0")

        return int(efs)

    @staticmethod
    def _normalize_attributes(attributes: Attributes) -> dict[str, Any] | None:
        if attributes is None:
            return None

        if not isinstance(attributes, Mapping):
            raise TypeError("attributes must be a mapping/dict")

        return dict(attributes)

    def __enter__(self) -> "ItemSearchEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
