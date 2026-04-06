import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tokenizers import Tokenizer

HEADER_SIZE = 5
ENCODING_VERSION = 1

_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def load_tokenizer(tokenizer_path: str) -> tuple[Tokenizer, int, int]:
    tokenizer = Tokenizer.from_file(tokenizer_path)

    pad_id = tokenizer.token_to_id("<PAD>")
    if pad_id is None:
        pad_id = 0

    unk_id = tokenizer.token_to_id("<UNK>")
    if unk_id is None:
        unk_id = -1

    return tokenizer, int(pad_id), int(unk_id)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold().strip()
    text = _WS_RE.sub(" ", text)
    return text


def _split_isolated_chunks(text: str) -> List[str]:

    text = _normalize_text(text)
    if not text:
        return []
    return [x for x in text.split(" ") if x]


def _token_ids_from_chunk(
    tokenizer: Tokenizer,
    chunk: str,
    pad_id: int,
    unk_id: int,
) -> List[int]:
    if not chunk:
        return []

    enc = tokenizer.encode(chunk, add_special_tokens=False)
    out = []
    for i in enc.ids:
        i = int(i)
        if i <= 0:
            continue
        if i == pad_id:
            continue
        if unk_id >= 0 and i == unk_id:
            continue
        out.append(i)
    return out


def _sorted_unique_chunk_token_ids(
    tokenizer: Tokenizer,
    text: str,
    pad_id: int,
    unk_id: int,
) -> List[int]:
    ids = set()
    for chunk in _split_isolated_chunks(text):
        ids.update(_token_ids_from_chunk(tokenizer, chunk, pad_id, unk_id))
    return sorted(ids)


def _iter_attr_strings(attributes: Optional[Dict[str, Any]]) -> Iterable[str]:
    if not attributes:
        return

    for raw_key, raw_value in sorted(attributes.items(), key=lambda kv: str(kv[0])):
        key = _normalize_text(str(raw_key))
        if not key or raw_value is None:
            continue

        values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
        seen_vals = set()

        for v in values:
            norm_v = _normalize_text(str(v))
            if not norm_v or norm_v in seen_vals:
                continue
            seen_vals.add(norm_v)
            yield f"{key}={norm_v}"


def _encode_title_ids(
    tokenizer: Tokenizer,
    title: str,
    pad_id: int,
    unk_id: int,
) -> List[int]:
    return _sorted_unique_chunk_token_ids(tokenizer, title, pad_id, unk_id)


def _encode_attr_ids(
    tokenizer: Tokenizer,
    attributes: Optional[Dict[str, Any]],
    pad_id: int,
    unk_id: int,
) -> List[int]:
    ids = set()
    for s in _iter_attr_strings(attributes):
        ids.update(_sorted_unique_chunk_token_ids(tokenizer, s, pad_id, unk_id))
    return sorted(ids)


def _encode_single_label_id(
    tokenizer: Tokenizer,
    text: str,
    pad_id: int,
    unk_id: int,
) -> int:
    """
    For now, keep brand/category as one canonical id:
    tokenize isolated chunks and use the first surviving token id.
    Later, a dedicated exact dictionary would be cleaner.
    """
    ids = _sorted_unique_chunk_token_ids(tokenizer, text, pad_id, unk_id)
    return int(ids[0]) if ids else 0


def encode_product_vector(
    tokenizer: Tokenizer,
    pad_id: int,
    unk_id: int,
    title: str,
    attributes: Optional[Dict[str, Any]],
    category: Optional[str],
    brand: Optional[str],
    max_dim: int,
) -> np.ndarray:
    if max_dim < HEADER_SIZE:
        raise ValueError(f"max_dim must be >= {HEADER_SIZE}, got {max_dim}")

    title_ids = _encode_title_ids(tokenizer, title or "", pad_id, unk_id)
    attr_ids = _encode_attr_ids(tokenizer, attributes, pad_id, unk_id)
    brand_id = _encode_single_label_id(tokenizer, brand or "", pad_id, unk_id)
    category_id = _encode_single_label_id(tokenizer, category or "", pad_id, unk_id)

    available = max_dim - HEADER_SIZE

    if len(title_ids) >= available:
        kept_title_ids = title_ids[:available]
        kept_attr_ids = []
    else:
        kept_title_ids = title_ids
        remaining = available - len(kept_title_ids)
        kept_attr_ids = attr_ids[:remaining]

    vec = np.zeros(max_dim, dtype=np.float32)
    vec[0] = float(ENCODING_VERSION)
    vec[1] = float(len(kept_title_ids))
    vec[2] = float(len(kept_attr_ids))
    vec[3] = float(brand_id)
    vec[4] = float(category_id)

    pos = HEADER_SIZE
    if kept_title_ids:
        vec[pos:pos + len(kept_title_ids)] = np.asarray(kept_title_ids, dtype=np.float32)
        pos += len(kept_title_ids)

    if kept_attr_ids:
        vec[pos:pos + len(kept_attr_ids)] = np.asarray(kept_attr_ids, dtype=np.float32)

    return vec


def encode_query_vector(
    tokenizer: Tokenizer,
    pad_id: int,
    unk_id: int,
    query: str,
    max_dim: int,
    attributes: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
) -> np.ndarray:
    if max_dim < HEADER_SIZE:
        raise ValueError(f"max_dim must be >= {HEADER_SIZE}, got {max_dim}")

    query_ids = _encode_title_ids(tokenizer, query or "", pad_id, unk_id)
    attr_ids = _encode_attr_ids(tokenizer, attributes, pad_id, unk_id)
    brand_id = _encode_single_label_id(tokenizer, brand or "", pad_id, unk_id)
    category_id = _encode_single_label_id(tokenizer, category or "", pad_id, unk_id)

    available = max_dim - HEADER_SIZE

    if len(query_ids) >= available:
        kept_query_ids = query_ids[:available]
        kept_attr_ids = []
    else:
        kept_query_ids = query_ids
        remaining = available - len(kept_query_ids)
        kept_attr_ids = attr_ids[:remaining]

    vec = np.zeros(max_dim, dtype=np.float32)
    vec[0] = float(ENCODING_VERSION)
    vec[1] = float(len(kept_query_ids))
    vec[2] = float(len(kept_attr_ids))
    vec[3] = float(brand_id)
    vec[4] = float(category_id)

    pos = HEADER_SIZE
    if kept_query_ids:
        vec[pos:pos + len(kept_query_ids)] = np.asarray(kept_query_ids, dtype=np.float32)
        pos += len(kept_query_ids)

    if kept_attr_ids:
        vec[pos:pos + len(kept_attr_ids)] = np.asarray(kept_attr_ids, dtype=np.float32)

    return vec

if __name__ == "__main__":
    products = [
        {
            "title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
            "attributes": {"color": "Natural Titanium", "storage": "256GB"},
            "category": "Smartphones",
            "brand": "Apple",
        },
        {
            "title": "Samsung Galaxy S24 Ultra",
            "attributes": {"color": "Black", "storage": "512GB", "sim": "Dual"},
            "category": "Smartphones",
            "brand": "Samsung",
        },
    ]

    tokenizer, pad_id = load_tokenizer("brinicle/pe_tokenizer_8k.json")
    # ....