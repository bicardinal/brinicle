from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class HNSWParams(BaseModel):
    M: Optional[int] = Field(
        default=16, description="Number of bi-directional links per node"
    )
    ef_construction: Optional[int] = Field(
        default=200, description="Size of dynamic candidate list during construction"
    )
    ef_search: Optional[int] = Field(
        default=64, description="Size of dynamic candidate list during search"
    )
    rng_seed: Optional[int] = Field(
        default=0, description="Random number generator seed"
    )


class CreateIndexRequest(BaseModel):
    index_name: str = Field(..., description="Unique name for the index")
    dim: int = Field(..., ge=1, description="Dimensionality of vectors")
    delta_ratio: float = Field(
        default=0.10, ge=0.0, le=0.5, description="Delta ratio parameter"
    )
    params: Optional[HNSWParams] = Field(default=None, description="HNSW parameters")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class LoadIndexRequest(BaseModel):
    index_name: str = Field(..., description="Unique name for the index")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class InitRequest(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    mode: Literal["build", "insert", "upsert"] = Field(
        ...,
        description="Ingest mode: 'build' for new index, 'insert' for adding, 'upsert' for update/insert",
    )

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class IngestRequest(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    external_id: str = Field(..., description="Unique identifier for the vector")
    vector: List[float] = Field(..., description="Vector data as list of floats")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        if not v:
            raise ValueError("Vector cannot be empty")
        return v

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class IngestBatchRequest(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    items: List[Dict[str, Any]] = Field(
        ..., description="List of {external_id, vector} dicts"
    )

    @field_validator("items")
    @classmethod
    def validate_items(cls, v):
        if not v:
            raise ValueError("Items list cannot be empty")
        for item in v:
            if "external_id" not in item or "vector" not in item:
                raise ValueError(
                    "Each item must contain 'external_id' and 'vector' fields"
                )
        return v

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class FinalizeRequest(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    params: Optional[HNSWParams] = Field(default=None, description="Build parameters")
    optimize: bool = Field(default=False, description="Whether to optimize after build")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class DeleteRequest(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    external_ids: List[str] = Field(..., description="List of external IDs to delete")
    return_not_found: bool = Field(
        default=False, description="Return list of IDs not found"
    )

    @field_validator("external_ids")
    @classmethod
    def validate_ids(cls, v):
        if not v:
            raise ValueError("external_ids list cannot be empty")
        return v

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class DeleteResponse(BaseModel):
    deleted_count: int = Field(..., description="Number of items deleted")
    not_found: Optional[List[str]] = Field(
        default=None, description="IDs that were not found"
    )


class RebuildRequest(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    params: Optional[HNSWParams] = Field(
        default=None, description="Build parameters for rebuild"
    )

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class IndexStatusResponse(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    dim: int = Field(..., description="Dimensionality of vectors")
    has_index: bool = Field(..., description="Whether an index exists")
    needs_rebuild: bool = Field(..., description="Whether index needs rebuild")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v):
        if v.replace("_", "").isalnum() and len(v) <= 64:
            return v
        raise ValueError(
            "index_name can contain A-Z|a-z|0-9 and underline(_) and should be <= 64 long."
        )


class ListIndexesResponse(BaseModel):
    indexes: List[str] = Field(..., description="List of index names")
    count: int = Field(..., description="Total number of indexes")


class SuccessResponse(BaseModel):
    success: bool = Field(default=True, description="Operation success status")
    message: Optional[str] = Field(default=None, description="Optional message")
    index_name: Optional[str] = Field(
        default=None, description="Index name for the operation"
    )
