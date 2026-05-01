from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HnswParams(BaseModel):
    M: int = 48
    ef_construction: int = 1024
    ef_search: int = 512
    rng_seed: int = 123


class CreateItemIndexRequest(BaseModel):
    index_name: str
    dim: int
    params: Optional[HnswParams] = None

    # Optional future controls.
    tokenizer_path: Optional[str] = None


class LoadItemIndexRequest(BaseModel):
    index_name: str
    dim: int
    params: Optional[HnswParams] = None
    # Optional future controls.
    tokenizer_path: Optional[str] = None

class InitItemRequest(BaseModel):
    index_name: str
    mode: str = "build"


class FinalizeItemRequest(BaseModel):
    index_name: str


class ItemDocument(BaseModel):
    id: str
    title: str
    category: Optional[str] = ""
    subcategory: Optional[str] = ""
    attributes: Optional[Dict] = None


class IngestItemRequest(BaseModel):
    index_name: str
    item: ItemDocument


class IngestItemBatchRequest(BaseModel):
    index_name: str
    items: List[ItemDocument]


class SearchItemRequest(BaseModel):
    index_name: str

    query: Optional[str] = None

    # For structured search later.
    title: Optional[str] = None
    category: Optional[str] = ""
    subcategory: Optional[str] = ""
    attributes: Optional[Dict] = None

    k: int = 10
    ef_search: Optional[int] = None


class SearchItemResult(BaseModel):
    id: str
    distance: Optional[float] = None
    score: Optional[float] = None


class SearchItemResponse(BaseModel):
    ids: List[str]
    results: Optional[List[SearchItemResult]] = None


class DeleteItemRequest(BaseModel):
    index_name: str
    external_ids: List[str]
    return_not_found: bool = False


class DeleteItemResponse(BaseModel):
    deleted_count: int
    not_found: Optional[List[str]] = None


class SuccessResponse(BaseModel):
    success: bool
    message: str
    index_name: Optional[str] = None


class ListIndexesResponse(BaseModel):
    indexes: List[str]
    count: int


class ItemIndexStatusResponse(BaseModel):
    index_name: str
    has_index: bool
    needs_rebuild: Optional[bool] = None
