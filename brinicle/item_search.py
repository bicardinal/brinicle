from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Union

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
        *,
        tokenizer_path: str | Path | None = None,
        text_prep: Any = None,
        title_ratio: float = 0.9,
        delta_ratio: float = 0.10,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 0,
        lexical_config: LexicalConfig | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be greater than 0")

        self.index_path = str(index_path)
        self._dim = int(dim)
        self._ef_search = int(ef_search)

        self.encoder = LexicalEncoder(
            tokenizer_path=tokenizer_path,
            max_dim=self._dim,
            text_prep=text_prep,
            title_ratio=title_ratio,
        )

        self.lexical_config = (
            lexical_config if lexical_config is not None else LexicalConfig()
        )

        self._engine = VectorEngine(
            self.index_path,
            self._dim,
            delta_ratio,
            M,
            ef_construction,
            ef_search,
            seed,
            "lexical",
            lexical_config=self.lexical_config,
        )

    def init(self, mode: str = "build") -> None:
        self._engine.init(mode)

    def ingest(
        self,
        external_id: str,
        title: str,
        category: Label = None,
        subcategory: Label = None,
        attributes: Attributes = None,
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
            normalize=normalize,
        )

        self._engine.ingest(str(external_id), self._as_f32(vec))

    def finalize(
        self,
        optimize: bool = False,
        M: int = 0,
        ef_construction: int = 0,
        ef_search: int = 0,
        seed: int = 0,
    ) -> None:
        self._engine.finalize(
            optimize=optimize,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
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
        *,
        normalize: bool = False,
    ) -> list[str]:

        qvec = self._encode_query(
            query=query,
            attributes=attributes,
            category=category,
            subcategory=subcategory,
            normalize=normalize,
        )

        return self._engine.search(
            self._as_f32(qvec),
            k=k,
            efs=self._resolve_efs(efs),
            threshold=threshold,
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
        *,
        normalize: bool = False,
    ) -> list[tuple[str, float]]:

        qvec = self._encode_query(
            query=query,
            attributes=attributes,
            category=category,
            subcategory=subcategory,
            normalize=normalize,
        )

        return self._engine.search_with_distance(
            self._as_f32(qvec),
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
        normalize: bool,
    ) -> np.ndarray:
        attrs = self._normalize_attributes(attributes)

        return self.encoder.encode_item_vector(
            title=title,
            attributes=attrs,
            category=category,
            subcategory=subcategory,
            normalize=normalize,
        )

    def _encode_query(
        self,
        *,
        query: str,
        attributes: Attributes,
        category: Label,
        subcategory: Label,
        normalize: bool,
    ) -> np.ndarray:
        attrs = self._normalize_attributes(attributes)
        return self.encoder.encode_query_vector(
            query=query,
            attributes=attrs,
            category=category,
            subcategory=subcategory,
            normalize=normalize,
        )

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
