from __future__ import annotations

from importlib.resources import as_file
from importlib.resources import files
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from tokenizers import Tokenizer


def _fnv1a_32(ids: List[int]) -> int:
    """
    FNV-1a 32-bit hash over a sorted list of integer token ids.
    Returns a value in [1, 2^23), safe for float32 exact representation.
    """
    FNV_OFFSET = 0x811C9DC5
    FNV_PRIME = 0x01000193
    h = FNV_OFFSET
    for x in ids:
        # feed each byte of the 32-bit id
        h ^= x & 0xFF
        h = (h * FNV_PRIME) & 0xFFFFFFFF
        h ^= (x >> 8) & 0xFF
        h = (h * FNV_PRIME) & 0xFFFFFFFF
        h ^= (x >> 16) & 0xFF
        h = (h * FNV_PRIME) & 0xFFFFFFFF
        h ^= (x >> 24) & 0xFF
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return (h & 0x7FFFFF) + 1  # 23-bit mask + 1, never zero


class TextPreprocess:
    def __init__(self):
        pass

    def normalize(self, text: str):
        return text


class LexicalEncoder:
    HEADER_SIZE = 6

    TITLE_TF_BITS = 4
    TITLE_TF_MASK = (1 << TITLE_TF_BITS) - 1
    TITLE_TF_MAX = TITLE_TF_MASK
    FLOAT32_EXACT_INT_MAX = (1 << 24) - 1

    special_token_names = (
        "<PAD>",
        "<UNK>",
        "<CLS>",
        "<SEP>",
        "<MASK>",
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    )

    def __init__(
        self,
        max_dim: int,
        vector_dim: int = -1,
        text_prep=None,
        tokenizer_path: Optional[str | Path] = None,
        title_ratio: float = 0.9,
    ):
        if not (0.0 < title_ratio <= 1.0):
            raise ValueError("title_ratio must be in (0, 1]")
        self.tokenizer, self.special_ids = self._load_tokenizer(tokenizer_path)

        self.vocab_size = self.tokenizer.get_vocab_size()

        if text_prep is None:
            self.text_prep = TextPreprocess()
        else:
            self.text_prep = text_prep
        self.max_dim = max_dim
        self.vector_dim = vector_dim
        self.title_ratio = title_ratio

        self.reserved_tokens = {True: self.vocab_size, False: self.vocab_size + 1}

        self.hash_namespace = self.vocab_size + len(self.reserved_tokens)

        if self.max_dim < self.HEADER_SIZE:
            raise ValueError(
                f"max_dim must be >= {self.HEADER_SIZE}, got {self.max_dim}"
            )

        payload = max_dim - self.HEADER_SIZE
        self.title_slots = int(payload * title_ratio)

        if self.title_slots < 1:
            raise ValueError("title_ratio too small, no room for title tokens")

    def _load_tokenizer(
        self,
        tokenizer_path: Optional[str | Path] = None,
    ) -> tuple[Tokenizer, set[int]]:

        def _collect_special_ids(tokenizer: Tokenizer) -> set[int]:
            return {
                int(tok_id)
                for tok in self.special_token_names
                if (tok_id := tokenizer.token_to_id(tok)) is not None
            }

        if tokenizer_path not in (None, ""):
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            return tokenizer, _collect_special_ids(tokenizer)

        resource = files("brinicle").joinpath("tokenizer.json")

        with as_file(resource) as tokenizer_file:
            tokenizer = Tokenizer.from_file(str(tokenizer_file))

        return tokenizer, _collect_special_ids(tokenizer)

    def _convert_num(self, the_num: int | float):
        the_num += self.vocab_size + len(self.reserved_tokens)
        return the_num

    def _norm_text(self, text: str) -> str:
        return self.text_prep.normalize(text)

    def _split_isolated_chunks(self, text: str) -> List[str]:
        if not text:
            return []
        return [x for x in text.split(" ") if x]

    def _token_ids_from_chunk(
        self,
        chunk: str,
    ) -> List[int]:
        if not chunk:
            return []

        enc = self.tokenizer.encode(chunk, add_special_tokens=False)
        out = []
        for i in enc.ids:
            if i > 0 and i not in self.special_ids:
                out.append(i)
        return out

    def _sorted_unique_chunk_token_ids(
        self,
        text: str,
    ) -> List[int]:
        ids = set()
        for chunk in text.split(" "):
            ids.update(self._token_ids_from_chunk(chunk))
        return sorted(ids)

    def _encode_title_ids(
        self,
        title: str,
    ) -> List[int]:
        return self._encode_title_token_ids_with_tf(title)

    def _pack_title_token_tf(self, token_id: int, tf: int) -> int:
        tf = max(1, min(int(tf), self.TITLE_TF_MAX))
        packed = (int(token_id) << self.TITLE_TF_BITS) | tf

        if packed > self.FLOAT32_EXACT_INT_MAX:
            raise ValueError(
                f"Packed title token id {packed} exceeds float32 exact integer range"
            )

        return packed

    def _encode_title_token_ids_with_tf(
        self,
        text: str,
    ) -> List[int]:
        """Our encoder should be deterministic, that is why we split the text before tokenizing."""
        freqs: dict[int, int] = {}
        for chunk in text.split(" "):
            for token_id in self._token_ids_from_chunk(chunk):
                freqs[token_id] = freqs.get(token_id, 0) + 1

        out = [
            self._pack_title_token_tf(token_id, tf) for token_id, tf in freqs.items()
        ][: self.title_slots]

        return sorted(out)

    def _hash_token_ids(self, ids: List[int]) -> int:
        """Single canonical id for a (possibly multi-token) field."""
        if not ids:
            return 0
        if len(ids) == 1:
            return ids[0]  # no hashing needed, already a clean id
        return _fnv1a_32(ids)

    def _encode_attributes(
        self,
        attributes: Optional[List | Dict],
    ) -> list[tuple[int, int]]:
        if not isinstance(attributes, dict) or not attributes:
            return []
        pairs = []
        for k, v in attributes.items():
            if k in (None, "", " ") or v in (None, "", " "):
                continue
            if not isinstance(k, str):
                raise ValueError(f"Unsupported attribute key type: {type(k)}")

            k = self._norm_text(k)
            k_id = self._hash_token_ids(self._sorted_unique_chunk_token_ids(k))
            if isinstance(v, bool):
                v_id = self.reserved_tokens[v]
            elif isinstance(v, (int, float)):
                v_id = self._convert_num(v)
            elif isinstance(v, str):
                v = self._norm_text(v)
                ids = self._sorted_unique_chunk_token_ids(v)
                v_id = self._hash_token_ids(ids)
            else:
                raise ValueError(f"Unsupported attribute value type: {type(v)}")
            pairs.append((k_id, v_id))
        pairs.sort()
        return pairs

    def _encode_label_id(self, label: str | int | None) -> int:
        if label is None:
            return 0
        if isinstance(label, int):
            return self.hash_namespace + label
        if not isinstance(label, str):
            raise ValueError(f"Unsupported label type: {type(label)}")
        label = self._norm_text(label)
        ids = self._sorted_unique_chunk_token_ids(label)
        if not ids:
            return 0
        return self.hash_namespace + _fnv1a_32(ids) if ids else 0

    def _encode_autocomplete_query(self, text: str):
        ids = []
        for chunk in text.split(" "):
            ids.extend(self._token_ids_from_chunk(chunk))
        return ids

    def encode_title_only_vector(
        self,
        title: str,
        normalize: bool = True,
    ) -> np.ndarray:
        if normalize:
            title = self._norm_text(title)

        title_ids = self._encode_title_token_ids_with_tf(title or "")

        vec = np.zeros(self.max_dim, dtype=np.float32)
        vec[0] = 0.0
        vec[1] = float(len(title_ids))
        vec[2] = 0.0
        vec[3] = 0.0
        vec[4] = 0.0

        if title_ids:
            n = len(title_ids)
            vec[self.HEADER_SIZE : self.HEADER_SIZE + n] = title_ids

        return vec

    def _build_vector(
        self,
        title: str,
        attributes: Optional[Dict[str, Any] | List | str],
        category: Optional[str | int],
        subcategory: Optional[str | int],
        vector: Optional[np.ndarray] = None,
        normalize: Optional[bool] = False,
    ) -> np.ndarray:
        vector_available = False
        if vector is not None:
            if not isinstance(vector, np.ndarray):
                raise ValueError("Invalid vector type")
            if vector.shape[0] != self.vector_dim:
                raise ValueError("Invalid vector shape")
            vector_available = True

        if not attributes and not category and not subcategory and not vector_available:
            return self.encode_title_only_vector(
                title=title,
                normalize=bool(normalize),
            )

        title = self._norm_text(title) if normalize else title
        title_ids = self._encode_title_ids(title or "")

        category = self._encode_label_id(category or "")
        subcategory = self._encode_label_id(subcategory or "")

        kept_attr_ids = []
        available = self.max_dim - self.HEADER_SIZE - len(title_ids)
        if available > 1:  # at least one pair
            kept_attr_ids = self._encode_attributes(attributes)[: int(available // 2)]

        vec = np.zeros(
            self.max_dim + self.vector_dim if vector_available else self.max_dim,
            dtype=np.float32,
        )
        vec[0] = 0  # later for version
        vec[1] = float(len(title_ids))
        vec[2] = float(len(kept_attr_ids))
        vec[3] = float(category)
        vec[4] = float(subcategory)
        vec[5] = float(self.vector_dim if vector_available else 0)

        pos = self.HEADER_SIZE
        if title_ids:
            vec[pos : pos + len(title_ids)] = np.asarray(title_ids, dtype=np.float32)
            pos += len(title_ids)

        if kept_attr_ids:
            for i, (k_hash, v_id) in enumerate(kept_attr_ids):
                vec[pos + i * 2] = float(k_hash)
                vec[pos + i * 2 + 1] = float(v_id)

        if vector_available:
            vec[self.max_dim :] = vector

        return vec

    def _build_autocomplete_vector(
        self,
        title: str,
        normalize: Optional[bool] = False,
    ) -> np.ndarray:
        title = self._norm_text(title) if normalize else title
        title_ids = self._encode_autocomplete_query(title or "")
        HEADER_SIZE = 1
        available = self.max_dim - HEADER_SIZE
        kept_title_ids = title_ids[:available]

        vec = np.zeros(self.max_dim, dtype=np.float32)
        vec[0] = float(len(kept_title_ids))

        pos = HEADER_SIZE
        if kept_title_ids:
            vec[pos : pos + len(kept_title_ids)] = np.asarray(
                kept_title_ids, dtype=np.float32
            )
            pos += len(kept_title_ids)

        return vec

    def encode_item_vector(
        self,
        title: str,
        attributes: Optional[Dict[str, Any]] = None,
        category: Optional[str | int] = None,
        subcategory: Optional[str | int] = None,
        vector: Optional[np.ndarray] = None,
        normalize: Optional[bool] = True,
    ) -> np.ndarray:
        return self._build_vector(
            title,
            attributes,
            category,
            subcategory,
            vector,
            normalize=normalize,
        )

    def encode_query_vector(
        self,
        query: str,
        attributes: Optional[Dict[str, Any]] = None,
        category: Optional[str | int] = None,
        subcategory: Optional[str | int] = None,
        vector: Optional[np.ndarray] = None,
        normalize: Optional[bool] = True,
    ) -> np.ndarray:
        return self._build_vector(
            query,
            attributes,
            category,
            subcategory,
            vector,
            normalize=normalize,
        )

    def encode_query_autocomplete_vector(
        self,
        query: str,
        normalize: Optional[bool] = True,
    ) -> np.ndarray:
        return self._build_autocomplete_vector(query, normalize=normalize)

    def encode_build_autocomplete_vector(
        self,
        query: str,
        normalize: Optional[bool] = True,
    ) -> np.ndarray:
        return self._build_autocomplete_vector(query, normalize=normalize)


if __name__ == "__main__":
    products = [
        {
            "title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
            "attributes": {"color": "Natural Titanium", "storage": "256GB"},
            "category": "Smartphones",
            "subcategory": "Apple",
        },
        {
            "title": "Samsung Galaxy S24 Ultra",
            "attributes": {"color": "Black", "storage": "512GB", "sim": "Dual"},
            "category": "Smartphones",
            "subcategory": "Samsung",
        },
    ]
    prep = TextPreprocess()
    lex = LexicalEncoder("examples/pe_tokenizer_4k.json", prep, 96)
    print(lex._encode_attributes(products[0]["attributes"]))
    print(lex._encode_attributes(products[1]["attributes"]))

    for product in products:
        print(
            lex.encode_item_vector(
                title=product["title"],
                attributes=product["attributes"],
                category=product["category"],
                subcategory=product["subcategory"],
            )
        )
    # ....
