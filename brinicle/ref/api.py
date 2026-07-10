"""
This wrapper is built for demo/benchmark purposes only.
Use the brinicle library directly instead.
"""

import traceback
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
import orjson
from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response

from brinicle._brinicle import VectorEngine
from brinicle.ref.io_models import CreateIndexRequest
from brinicle.ref.io_models import DeleteRequest
from brinicle.ref.io_models import DeleteResponse
from brinicle.ref.io_models import FinalizeRequest
from brinicle.ref.io_models import IndexStatusResponse
from brinicle.ref.io_models import InitRequest
from brinicle.ref.io_models import ListIndexesResponse
from brinicle.ref.io_models import LoadIndexRequest
from brinicle.ref.io_models import RebuildRequest
from brinicle.ref.io_models import SuccessResponse

indexes: Dict[str, VectorEngine] = {}
store_dir = "/app/data/"

MAX_SEARCH_BATCH_SIZE = 8192


@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexes

    yield

    for index_name, engine in indexes.items():
        try:
            engine.close()
            print(f"Closed index: {index_name}")
        except Exception as e:
            print(f"Error closing index {index_name}: {e}")


app = FastAPI(
    title="Vector Engine API",
    description="FastAPI wrapper for HNSW vector search engine",
    version="0.0.0",
    lifespan=lifespan,
)


def get_engine(index_name: str):
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    return indexes[index_name]


@app.get("/", response_model=SuccessResponse)
async def root():
    return SuccessResponse(
        success=True,
        message=f"Vector Engine API is running. {len(indexes)} index(es) loaded.",
    )


@app.get("/indexes", response_model=ListIndexesResponse)
async def list_indexes():
    return ListIndexesResponse(indexes=list(indexes.keys()), count=len(indexes))


@app.post("/indexes", response_model=SuccessResponse)
async def create_index(request: CreateIndexRequest):
    if request.index_name in indexes:
        raise HTTPException(
            status_code=409, detail=f"Index '{request.index_name}' already exists"
        )

    if VectorEngine is None:
        raise HTTPException(status_code=503, detail="VectorEngine module not available")

    params = request.params
    try:
        if params:
            engine = VectorEngine(
                store_dir + request.index_name,
                request.dim,
                request.delta_ratio,
                params.M,
                params.ef_construction,
                params.ef_search,
                build_n_threads=params.build_n_threads,
                seed=params.rng_seed,
            )
            print(params)
        else:
            engine = VectorEngine(
                store_dir + request.index_name,
                request.dim,
                request.delta_ratio,
            )

        indexes[request.index_name] = engine

        return SuccessResponse(
            success=True,
            message=f"Index '{request.index_name}' created successfully",
            index_name=request.index_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to create index: {str(e)}")


@app.delete("/indexes/{index_name}", response_model=SuccessResponse)
async def delete_index(index_name: str, destroy: bool = False):
    engine = get_engine(index_name)

    try:
        if destroy:
            engine.destroy()
        else:
            engine.close()

        del indexes[index_name]

        action = "destroyed" if destroy else "closed and removed"
        return SuccessResponse(
            success=True,
            message=f"Index '{index_name}' {action}",
            index_name=index_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to delete index: {str(e)}")


@app.get("/indexes/{index_name}/status", response_model=IndexStatusResponse)
async def get_index_status(index_name: str):
    engine = get_engine(index_name)

    return IndexStatusResponse(
        index_name=index_name,
        dim=engine.dim,
        has_index=engine.has_index,
        needs_rebuild=engine.needs_rebuild(),
    )


@app.post("/indexes/load", response_model=SuccessResponse)
async def load_index(request: LoadIndexRequest):
    index_name = request.index_name

    # if index_name in indexes:
    #     raise HTTPException(
    #         status_code=409,
    #         detail=f"Index '{index_name}' is already loaded",
    #     )

    engine = VectorEngine(store_dir + index_name, 0)
    indexes[index_name] = engine

    return SuccessResponse(
        success=True,
        message=f"Index '{index_name}' loaded successfully",
        index_name=index_name,
    )


@app.post("/init", response_model=SuccessResponse)
async def initialize_ingest(request: InitRequest):
    engine = get_engine(request.index_name)

    try:
        engine.init(request.mode)
        return SuccessResponse(
            success=True,
            message=f"Index '{request.index_name}' initialized in '{request.mode}' mode",
            index_name=request.index_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to initialize: {str(e)}")


@app.post("/ingest")
async def ingest_single(request: Request):
    try:
        body = await request.body()
        data = orjson.loads(body)
        index_name = data.get("index_name")
        external_id = data.get("external_id")
        vector = data.get("vector")
        if not index_name or not external_id or vector is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: index_name, external_id, vector",
            )

        engine = get_engine(index_name)

        # Convert directly to numpy and ingest immediately
        vec_array = np.array(vector, dtype=np.float32)
        engine.ingest(external_id, vec_array)

        # Return minimal response
        return Response(content=b'{"success":true}', media_type="application/json")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/batch")
async def ingest_batch(request: Request):
    try:
        index_name = request.query_params.get("index_name")
        if not index_name:
            raise HTTPException(
                status_code=400, detail="Missing query param: index_name"
            )

        engine = get_engine(index_name)
        dim = engine.dim

        id_bytes = 32
        vec_bytes = dim * 4
        row_bytes = id_bytes + vec_bytes

        carry = b""
        batch_count = 0

        async for chunk in request.stream():
            if not chunk:
                continue

            chunk = carry + chunk

            num_complete_rows = len(chunk) // row_bytes
            bytes_to_process = num_complete_rows * row_bytes

            if bytes_to_process > 0:
                carry = chunk[bytes_to_process:]
                chunk = chunk[:bytes_to_process]

                offset = 0
                while offset < bytes_to_process:
                    id_bytes_data = chunk[offset : offset + id_bytes]
                    external_id = id_bytes_data.decode("ascii").rstrip("\x00")
                    offset += id_bytes

                    vec_bytes_data = chunk[offset : offset + vec_bytes]
                    vec_array = np.frombuffer(vec_bytes_data, dtype=np.float32).copy()
                    offset += vec_bytes

                    engine.ingest(external_id, vec_array)
                    batch_count += 1

            else:
                carry = chunk
        if carry:
            raise HTTPException(
                status_code=400,
                detail=f"Ingest payload not aligned to rows: leftover {len(carry)} bytes, expected multiples of {row_bytes}",
            )
        return Response(
            content=orjson.dumps({"success": True, "count": batch_count}),
            media_type="application/json",
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/finalize", response_model=SuccessResponse)
async def finalize_ingest(request: FinalizeRequest):
    engine = get_engine(request.index_name)

    try:
        params = request.params
        if params:
            engine.finalize(
                request.optimize,
                params.M,
                params.ef_construction,
                params.ef_search,
                params.rng_seed,
            )
        else:
            engine.finalize(request.optimize)
        return SuccessResponse(
            success=True,
            message=f"Finalized ingest for index '{request.index_name}' (optimize={request.optimize})",
            index_name=request.index_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to finalize: {str(e)}")


@app.post("/delete", response_model=DeleteResponse)
async def delete_items(request: DeleteRequest):
    engine = get_engine(request.index_name)

    try:
        deleted_count, not_found = engine.delete_items(
            request.external_ids, request.return_not_found
        )

        return DeleteResponse(
            deleted_count=deleted_count,
            not_found=not_found if request.return_not_found else None,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to delete: {str(e)}")


@app.post("/rebuild", response_model=SuccessResponse)
async def rebuild_compact(request: RebuildRequest):
    engine = get_engine(request.index_name)

    try:
        params = request.params
        if params:
            engine.rebuild_compact(
                params.M,
                params.ef_construction,
                params.ef_search,
                params.rng_seed,
            )
        else:
            engine.rebuild_compact()
        return SuccessResponse(
            success=True,
            message=f"Index '{request.index_name}' rebuilt and compacted",
            index_name=request.index_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to rebuild: {str(e)}")


@app.post("/search.bin")
async def search_bin(
    index_name: str,
    k: int = 10,
    efs: int = 64,
    body: bytes = Body(..., media_type="application/octet-stream"),
):
    engine = get_engine(index_name)

    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")
    if efs <= 0:
        raise HTTPException(status_code=400, detail="efs must be greater than 0")

    query = np.frombuffer(body, dtype=np.float32)
    if query.size != engine.dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected one float32 vector with dim={engine.dim}, got {query.size} values",
        )

    neighbors = engine.search(query, k=k, efs=efs)
    return Response(
        content=orjson.dumps(neighbors, option=orjson.OPT_SERIALIZE_NUMPY),
        media_type="application/json",
    )


@app.post("/search/batch.bin")
async def search_batch_bin(
    index_name: str,
    k: int = 10,
    efs: int = 64,
    n_jobs: int = 1,
    body: bytes = Body(..., media_type="application/octet-stream"),
):
    """Run real VectorEngine batch search over raw float32 query vectors.

    Request body layout:
      [query_0 float32 dim values][query_1 float32 dim values]...

    Query params:
      index_name: loaded index name
      k: top-k result count
      efs: search ef / query-time search width
      n_jobs: worker count passed to VectorEngine.search_batch; use -1 for engine default/all
    """

    engine = get_engine(index_name)

    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")
    if efs <= 0:
        raise HTTPException(status_code=400, detail="efs must be greater than 0")
    if n_jobs == 0:
        raise HTTPException(
            status_code=400,
            detail="n_jobs must be 1, greater than 1, or -1.",
        )

    dim = int(engine.dim)
    if dim <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Index '{index_name}' has invalid dim={dim}",
        )

    row_bytes = dim * 4
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Empty batch search body")
    if len(body) % row_bytes != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch search payload is not aligned to float32 rows: "
                f"got {len(body)} bytes, expected a multiple of {row_bytes} bytes "
                f"for dim={dim}."
            ),
        )

    batch_size = len(body) // row_bytes
    if batch_size > MAX_SEARCH_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch too large. Max allowed batch size is "
                f"{MAX_SEARCH_BATCH_SIZE}, got {batch_size}."
            ),
        )

    queries = np.frombuffer(body, dtype=np.float32).reshape(batch_size, dim)

    ids_by_query = engine.search_batch(queries, k=k, efs=efs, n_jobs=n_jobs)

    if len(ids_by_query) != batch_size:
        raise RuntimeError(
            f"Batch search returned {len(ids_by_query)} result lists "
            f"for {batch_size} queries."
        )

    return Response(
        content=orjson.dumps(ids_by_query, option=orjson.OPT_SERIALIZE_NUMPY),
        media_type="application/json",
    )


@app.post("/optimize", response_model=SuccessResponse)
async def optimize_graph(index_name: str = Body(..., description="Name of the index")):
    engine = get_engine(index_name)

    try:
        engine.optimize_graph()
        return SuccessResponse(
            success=True,
            message=f"Graph optimized for index '{index_name}'",
            index_name=index_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to optimize: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1984)
