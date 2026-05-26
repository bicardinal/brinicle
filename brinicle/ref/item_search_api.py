"""
This wrapper is built for demo/benchmark purposes only.
Use the brinicle library directly instead.
"""

import traceback
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
import orjson
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response

from brinicle import ItemSearchEngine
from brinicle.ref.item_io_models import CreateItemIndexRequest
from brinicle.ref.item_io_models import DeleteItemRequest
from brinicle.ref.item_io_models import DeleteItemResponse
from brinicle.ref.item_io_models import FinalizeItemRequest
from brinicle.ref.item_io_models import InitItemRequest
from brinicle.ref.item_io_models import ItemIndexStatusResponse
from brinicle.ref.item_io_models import ListIndexesResponse
from brinicle.ref.item_io_models import LoadItemIndexRequest
from brinicle.ref.item_io_models import SuccessResponse

indexes: Dict[str, ItemSearchEngine] = {}
store_dir = "/app/data/"

MAX_SEARCH_BATCH_SIZE = 8192
MAX_INGEST_LINE_BYTES = 32 * 1024 * 1024  # 32 MB safety guard


@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexes
    yield

    for index_name, engine in indexes.items():
        try:
            engine.close()
            print(f"Closed item index: {index_name}")
        except Exception as e:
            print(f"Error closing item index {index_name}: {e}")


app = FastAPI(
    title="Brinicle Item Search API",
    description="FastAPI wrapper for Brinicle ItemSearchEngine",
    version="0.0.0",
    lifespan=lifespan,
)


def get_engine(index_name: str) -> ItemSearchEngine:
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    return indexes[index_name]


def _ingest_item_dict(engine: ItemSearchEngine, item: dict):
    external_id = item.get("id") or item.get("external_id")

    if external_id is None:
        raise ValueError("Item is missing id/external_id")

    title = item.get("title") or ""

    kwargs = {
        "external_id": str(external_id),
        "title": str(title),
        "category": item.get("category") or "",
        "subcategory": item.get("subcategory") or "",
        "attributes": item.get("attributes"),
    }

    vector = item.get("vector")
    if vector is not None:
        kwargs["vector"] = np.asarray(vector, dtype=np.float32)

    engine.ingest(**kwargs)


@app.get("/", response_model=SuccessResponse)
async def root():
    return SuccessResponse(
        success=True,
        message=f"Brinicle Item Search API is running. {len(indexes)} index(es) loaded.",
    )


@app.get("/indexes", response_model=ListIndexesResponse)
async def list_indexes():
    return ListIndexesResponse(indexes=list(indexes.keys()), count=len(indexes))


@app.post("/indexes", response_model=SuccessResponse)
async def create_index(request: CreateItemIndexRequest):
    index_name = request.index_name
    old_engine = indexes.get(index_name)
    if old_engine is not None:
        try:
            old_engine.close()
        except Exception as e:
            print(f"Error closing previous item index {index_name}: {e}")

    try:
        params = request.params
        path = store_dir + index_name

        if params:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                vector_dim=request.vector_dim,
                M=params.M,
                ef_construction=params.ef_construction,
                ef_search=params.ef_search,
                build_n_threads=params.build_n_threads,
                alpha=params.alpha,
                seed=params.seed,
                tokenizer_path=request.tokenizer_path,
                vector_normalized=request.vector_normalized,
            )
        else:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                vector_dim=request.vector_dim,
                tokenizer_path=request.tokenizer_path,
                vector_normalized=request.vector_normalized,
            )
        old_engine = indexes.get(index_name)
        if old_engine is not None:
            try:
                old_engine.close()
            except Exception as e:
                print(f"Error closing previous item index {index_name}: {e}")

        indexes[index_name] = engine

        return SuccessResponse(
            success=True,
            message=f"Item index '{index_name}' created successfully",
            index_name=index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to create item index: {str(e)}"
        )


@app.delete("/indexes/{index_name}", response_model=SuccessResponse)
async def delete_index(index_name: str, destroy: bool = False):
    engine = get_engine(index_name)

    try:
        if destroy and hasattr(engine, "destroy"):
            engine.destroy()
        else:
            engine.close()

        del indexes[index_name]

        action = "destroyed" if destroy else "closed and removed"
        return SuccessResponse(
            success=True,
            message=f"Item index '{index_name}' {action}",
            index_name=index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to delete item index: {str(e)}"
        )


@app.get("/indexes/{index_name}/status", response_model=ItemIndexStatusResponse)
async def get_index_status(index_name: str):
    engine = get_engine(index_name)

    needs_rebuild = None
    if hasattr(engine, "needs_rebuild"):
        needs_rebuild = engine.needs_rebuild()

    return ItemIndexStatusResponse(
        index_name=index_name,
        has_index=getattr(engine, "has_index", True),
        needs_rebuild=needs_rebuild,
    )


@app.post("/indexes/load", response_model=SuccessResponse)
async def load_index(request: LoadItemIndexRequest):
    try:
        index_name = request.index_name
        params = request.params
        path = store_dir + request.index_name

        if params:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                vector_dim=request.vector_dim,
                M=params.M,
                ef_construction=params.ef_construction,
                ef_search=params.ef_search,
                alpha=params.alpha,
                build_n_threads=params.build_n_threads,
                seed=params.seed,
                tokenizer_path=request.tokenizer_path,
                vector_normalized=request.vector_normalized,
            )
        else:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                vector_dim=request.vector_dim,
                tokenizer_path=request.tokenizer_path,
                vector_normalized=request.vector_normalized,
            )

        old_engine = indexes.get(index_name)
        if old_engine is not None:
            try:
                old_engine.close()
            except Exception as e:
                print(f"Error closing previous item index {index_name}: {e}")
        indexes[index_name] = engine

        return SuccessResponse(
            success=True,
            message=f"Item index '{index_name}' loaded successfully",
            index_name=index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to load item index: {str(e)}"
        )


@app.post("/init", response_model=SuccessResponse)
async def initialize_ingest(request: InitItemRequest):
    engine = get_engine(request.index_name)

    try:
        engine.init(request.mode)

        return SuccessResponse(
            success=True,
            message=f"Item index '{request.index_name}' initialized in '{request.mode}' mode",
            index_name=request.index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to initialize item index: {str(e)}"
        )


@app.post("/ingest")
async def ingest_single(request: Request):
    try:
        payload = await request.json()

        index_name = payload.get("index_name")
        if not index_name:
            raise HTTPException(status_code=400, detail="Missing index_name")

        item = payload.get("item")
        if not item:
            raise HTTPException(status_code=400, detail="Missing item")

        engine = get_engine(index_name)
        _ingest_item_dict(engine, item)

        return Response(
            content=b'{"success":true,"count":1}',
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to ingest item: {str(e)}")




@app.post("/ingest/batch")
async def ingest_batch(request: Request):

    try:
        index_name = request.query_params.get("index_name")
        if not index_name:
            raise HTTPException(
                status_code=400,
                detail="Missing query param: index_name",
            )

        engine = get_engine(index_name)

        carry = b""
        count = 0

        async for chunk in request.stream():
            if not chunk:
                continue

            carry += chunk

            if len(carry) > MAX_INGEST_LINE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Ingest line too large. Current carry={len(carry)} bytes.",
                )

            while True:
                newline_pos = carry.find(b"\n")
                if newline_pos < 0:
                    break

                line = carry[:newline_pos].strip()
                carry = carry[newline_pos + 1 :]

                if not line:
                    continue

                item = orjson.loads(line)
                _ingest_item_dict(engine, item)
                count += 1

        if carry.strip():
            item = orjson.loads(carry)
            _ingest_item_dict(engine, item)
            count += 1

        return Response(
            content=orjson.dumps({"success": True, "count": count}),
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to batch ingest items: {str(e)}",
        )


@app.post("/finalize", response_model=SuccessResponse)
async def finalize_ingest(request: FinalizeItemRequest):
    engine = get_engine(request.index_name)

    try:
        engine.finalize()
        return SuccessResponse(
            success=True,
            message=f"Finalized item ingest for index '{request.index_name}'",
            index_name=request.index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to finalize item index: {str(e)}"
        )


@app.post("/search")
async def search_items(request: Request):
    try:
        payload = await request.json()

        index_name = payload.get("index_name")
        if not index_name:
            raise HTTPException(status_code=400, detail="Missing index_name")

        engine = get_engine(index_name)

        query = str(payload.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Missing query")

        k = int(payload.get("k", 10))

        kwargs = {
            "query": query,
            "category": payload.get("category") or "",
            "subcategory": payload.get("subcategory") or "",
            "attributes": payload.get("attributes"),
            "k": k,
        }

        ef_search = payload.get("ef_search")
        if ef_search is not None:
            kwargs["efs"] = int(ef_search)

        vector = payload.get("vector")
        if vector is not None:
            kwargs["vector"] = np.asarray(vector, dtype=np.float32)

        ids = engine.search(**kwargs)
        return ids

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to search item index: {str(e)}",
        )


@app.post("/search/batch")
async def search_items_batch(request: Request):
    try:
        payload = await request.json()

        index_name = payload.get("index_name")
        engine = get_engine(index_name)

        queries = payload.get("queries") or []
        queries = [str(q or "").strip() for q in queries]

        if len(queries) > MAX_SEARCH_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Batch too large. Max allowed batch size is "
                    f"{MAX_SEARCH_BATCH_SIZE}, got {len(queries)}."
                ),
            )

        k = int(payload.get("k", 10))
        n_jobs = int(payload.get("n_jobs", 1))
        if n_jobs == 0:
            raise HTTPException(
                status_code=400,
                detail="n_jobs must be 1, greater than 1, or -1.",
            )

        ef_search = payload.get("ef_search")
        if ef_search is not None and int(ef_search) <= 0:
            raise HTTPException(
                status_code=400,
                detail="ef_search must be greater than 0.",
            )

        kwargs = {
            "queries": queries,
            "category": payload.get("category") or "",
            "subcategory": payload.get("subcategory") or "",
            "attributes": payload.get("attributes"),
            "k": k,
            "n_jobs": n_jobs,
        }

        if ef_search is not None:
            kwargs["efs"] = int(ef_search)

        vectors = payload.get("vectors")
        if vectors is not None:
            kwargs["vectors"] = np.asarray(vectors, dtype=np.float32)

        ids_by_query = engine.search_batch(**kwargs)

        if len(ids_by_query) != len(queries):
            raise RuntimeError(
                f"Batch search returned {len(ids_by_query)} result lists "
                f"for {len(queries)} queries."
            )

        return ids_by_query

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to batch search item index: {str(e)}",
        )


@app.post("/search_with_distance")
async def search_with_distance(request: Request):
    try:
        payload = await request.json()

        index_name = payload.get("index_name")
        if not index_name:
            raise HTTPException(status_code=400, detail="Missing index_name")

        engine = get_engine(index_name)

        query = str(payload.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Missing query")

        k = int(payload.get("k", 10))

        kwargs = {
            "query": query,
            "category": payload.get("category") or "",
            "subcategory": payload.get("subcategory") or "",
            "attributes": payload.get("attributes"),
            "k": k,
        }

        ef_search = payload.get("ef_search")
        if ef_search is not None:
            kwargs["efs"] = int(ef_search)

        vector = payload.get("vector")
        if vector is not None:
            kwargs["vector"] = np.asarray(vector, dtype=np.float32)

        ids = engine.search_with_distance(**kwargs)
        return ids

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to search item index: {str(e)}",
        )


@app.post("/delete", response_model=DeleteItemResponse)
async def delete_items(request: DeleteItemRequest):
    engine = get_engine(request.index_name)

    try:
        deleted_count, not_found = engine.delete_items(
            request.external_ids,
            request.return_not_found,
        )

        return DeleteItemResponse(
            deleted_count=deleted_count,
            not_found=not_found if request.return_not_found else None,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to delete items: {str(e)}")


@app.post("/rebuild", response_model=SuccessResponse)
async def rebuild_compact(request: FinalizeItemRequest):
    engine = get_engine(request.index_name)

    try:
        params = request.params

        if not hasattr(engine, "rebuild_compact"):
            raise HTTPException(
                status_code=501,
                detail="ItemSearchEngine does not expose rebuild_compact",
            )

        if params:
            engine.rebuild_compact(
                M=params.M,
                ef_construction=params.ef_construction,
                ef_search=params.ef_search,
                n_threads=params.n_threads,
                batch_size=params.batch_size,
                seed=params.seed,
            )
        else:
            engine.rebuild_compact()

        return SuccessResponse(
            success=True,
            message=f"Item index '{request.index_name}' rebuilt and compacted",
            index_name=request.index_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to rebuild item index: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1984)
