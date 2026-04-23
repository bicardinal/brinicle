from __future__ import annotations

import math, re, unicodedata, json

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any
from collections import Counter, defaultdict
import numpy as np
from tokenizers import Tokenizer

def _fnv1a_32(ids: List[int]) -> int:
	"""
	FNV-1a 32-bit hash over a sorted list of integer token ids.
	Returns a value in [1, 2^23), safe for float32 exact representation.
	"""
	FNV_OFFSET = 0x811C9DC5
	FNV_PRIME  = 0x01000193
	h = FNV_OFFSET
	for x in ids:
		# feed each byte of the 32-bit id
		h ^= (x & 0xFF);         h = (h * FNV_PRIME) & 0xFFFFFFFF
		h ^= ((x >> 8)  & 0xFF); h = (h * FNV_PRIME) & 0xFFFFFFFF
		h ^= ((x >> 16) & 0xFF); h = (h * FNV_PRIME) & 0xFFFFFFFF
		h ^= ((x >> 24) & 0xFF); h = (h * FNV_PRIME) & 0xFFFFFFFF
	return (h & 0x7FFFFF) + 1  # 23-bit mask + 1, never zero


class TextPreprocess:
	def __init__(self):
		pass

	def normalize(self, text: str):
		return text

class LexicalEncoder:
	HEADER_SIZE = 5

	def __init__(
		self,
		tokenizer_path: str,
		max_dim: int,
		text_prep = None,
		title_ratio: float = 0.9,
	):
		if not (0.0 < title_ratio <= 1.0):
			raise ValueError("title_ratio must be in (0, 1]")
		tokenizer, pad_id, unk_id = self._load_tokenizer(tokenizer_path)
		self.tokenizer = tokenizer
		self.pad_id = pad_id
		self.unk_id = unk_id
		self.vocab_size = tokenizer.get_vocab_size()
		
		if text_prep is None:
			self.text_prep = TextPreprocess()
		else:
			self.text_prep = text_prep
		self.max_dim = max_dim
		self.title_ratio = title_ratio

		self.reserved_tokens = {
			True: self.vocab_size,
			False: self.vocab_size + 1
		}

		if self.max_dim < self.HEADER_SIZE:
			raise ValueError(f"max_dim must be >= {self.HEADER_SIZE}, got {self.max_dim}")

		payload = max_dim - self.HEADER_SIZE
		self.title_slots = int(payload * title_ratio)

		if self.title_slots < 1:
			raise ValueError("title_ratio too small, no room for title tokens")


	def _load_tokenizer(self, tokenizer_path: str) -> tuple[Tokenizer, int, int]:
		tokenizer = Tokenizer.from_file(tokenizer_path)
		pad_id = tokenizer.token_to_id("<PAD>")
		if pad_id is None:
			pad_id = 0
		unk_id = tokenizer.token_to_id("<UNK>")
		if unk_id is None:
			unk_id = -1
		return tokenizer, int(pad_id), int(unk_id)

	def _convert_num(self, the_num: int|float):
		the_num += self.vocab_size + len(self.reserved_tokens)
		return the_num

	def _norm_text(self, text: str) -> str:
		return self.text_prep.normalize(text)

	def _split_isolated_chunks(self, text: str) -> List[str]:
		# text = self._norm_text(text)
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
			i = int(i)
			if i <= 0:
				continue
			if i == self.pad_id:
				continue
			if self.unk_id >= 0 and i == self.unk_id:
				continue
			out.append(i)
		return out

	def _sorted_unique_chunk_token_ids(
		self,
		text: str,
	) -> List[int]:
		ids = set()
		for chunk in self._split_isolated_chunks(text):
			ids.update(self._token_ids_from_chunk(chunk))
		return sorted(ids)

	def _encode_title_ids(
		self,
		title: str,
	) -> List[int]:
		return self._sorted_unique_chunk_token_ids(title)

	def _hash_token_ids(self, ids: List[int]) -> int:
		"""Single canonical id for a (possibly multi-token) field."""
		if not ids:
			return 0
		if len(ids) == 1:
			return ids[0] # no hashing needed, already a clean id
		return _fnv1a_32(ids)

	def _encode_attributes(
		self,
		attributes: Optional[List|Dict],
	) -> List[int]:
		if not isinstance(attributes, dict) or not attributes:
			return []
		pairs = []
		for k,v in attributes.items():
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

	def _encode_single_label_id(
		self,
		text: str,
	) -> int:
		text = self._norm_text(text)
		ids = self._sorted_unique_chunk_token_ids(text)
		return int(ids[0]) if ids else 0

	def _encode_autocomplete_query(self, text: str):
		ids = []
		for chunk in self._split_isolated_chunks(text):
			ids.extend(self._token_ids_from_chunk(chunk))
		return ids

	def _build_vector(
		self,
		title: str,
		attributes: Optional[Dict[str, Any]|List|str],
		category_id: Optional[str|int],
		brand_id: Optional[str|int],
		dont_normalize: Optional[bool] = False,
	) -> np.ndarray:
		title = title if dont_normalize else self._norm_text(title)
		title_ids = self._encode_title_ids(title or "")
		kept_title_ids = title_ids[:self.title_slots]
		if isinstance(brand_id, str) or brand_id is None:
			brand_id = self._encode_single_label_id(brand_id or "")
		if isinstance(category_id, str) or category_id is None:
			category_id = self._encode_single_label_id(category_id or "")

		kept_attr_ids = []
		available = self.max_dim - self.HEADER_SIZE - len(kept_title_ids)
		if available > 1: # at least one pair
			kept_attr_ids = self._encode_attributes(attributes)[:int(available // 2)]

		vec = np.zeros(self.max_dim, dtype=np.float32)
		vec[0] = 0
		vec[1] = float(len(kept_title_ids))
		vec[2] = float(len(kept_attr_ids))
		vec[3] = float(brand_id)
		vec[4] = float(category_id)

		pos = self.HEADER_SIZE
		if kept_title_ids:
			vec[pos:pos + len(kept_title_ids)] = np.asarray(kept_title_ids, dtype=np.float32)
			pos += len(kept_title_ids)

		if kept_attr_ids:
			for i, (k_hash, v_id) in enumerate(kept_attr_ids):
				vec[pos + i * 2] = float(k_hash)
				vec[pos + i * 2 + 1] = float(v_id)
		return vec

	def _build_autocomplete_vector(
		self,
		title: str,
		dim: int,
		dont_normalize: Optional[bool] = False,
	) -> np.ndarray:
		title = title if dont_normalize else self._norm_text(title)
		title_ids = self._encode_autocomplete_query(title or "")
		HEADER_SIZE = 1
		available = dim - HEADER_SIZE
		kept_title_ids = title_ids[:available]

		vec = np.zeros(self.max_dim, dtype=np.float32)
		vec[0] = float(len(kept_title_ids))

		pos = HEADER_SIZE
		if kept_title_ids:
			vec[pos:pos + len(kept_title_ids)] = np.asarray(kept_title_ids, dtype=np.float32)
			pos += len(kept_title_ids)

		return vec

	def encode_product_vector(
		self,
		title: str,
		attributes: Optional[Dict[str, Any]] = None,
		category_id: Optional[str|int] = None,
		brand_id: Optional[str|int] = None,
		dont_normalize: Optional[bool] = False,
	) -> np.ndarray:
		return self._build_vector(title, attributes, category_id, brand_id, dont_normalize=dont_normalize)

	def encode_query_vector(
		self,
		query: str,
		attributes: Optional[Dict[str, Any]] = None,
		category_id: Optional[str|int] = None,
		brand_id: Optional[str|int] = None,
		dont_normalize: Optional[bool] = False,
	) -> np.ndarray:
		return self._build_vector(query, attributes, category_id, brand_id, dont_normalize=dont_normalize)


	def encode_query_autocomplete_vector(
		self,
		query: str,
		dim: int,
		dont_normalize: Optional[bool] = False,
	) -> np.ndarray:
		return self._build_autocomplete_vector(query, dim, dont_normalize=dont_normalize)

	def encode_build_autocomplete_vector(
		self,
		query: str,
		dim: int,
		dont_normalize: Optional[bool] = False,
	) -> np.ndarray:
		return self._build_autocomplete_vector(query, dim, dont_normalize=dont_normalize)




if __name__ == "__main__":
	products = [
		{
			"title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
			"attributes": {"color": "Natural Titanium", "storage": "256GB"},
			"category_id": "Smartphones",
			"brand_id": "Apple",
		},
		{
			"title": "Samsung Galaxy S24 Ultra",
			"attributes": {"color": "Black", "storage": "512GB", "sim": "Dual"},
			"category_id": "Smartphones",
			"brand_id": "Samsung",
		},
	]
	prep = TextPreprocess()
	lex = LexicalEncoder("examples/pe_tokenizer_4k.json", prep, 96)
	print(lex._encode_attributes(products[0]["attributes"]))
	print(lex._encode_attributes(products[1]["attributes"]))

	for product in products:
		print(lex.encode_product_vector(
			title=product["title"],
			attributes=product["attributes"],
			category_id=product["category_id"],
			brand_id=product["brand_id"],
		))
	# ....
