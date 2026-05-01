"""
This wrapper is built for demo/benchmark purposes only.
Use the brinicle library directly instead.
"""

import traceback
from contextlib import asynccontextmanager
from typing import Dict, List, Any

import orjson
from fastapi import FastAPI, HTTPException, Request, Response

from brinicle import ItemSearchEngine

from brinicle.ref.item_io_models import (
    CreateItemIndexRequest,
    LoadItemIndexRequest,
    InitItemRequest,
    FinalizeItemRequest,
    IngestItemRequest,
    IngestItemBatchRequest,
    SearchItemRequest,
    SearchItemResponse,
    SearchItemResult,
    DeleteItemRequest,
    DeleteItemResponse,
    SuccessResponse,
    ListIndexesResponse,
    ItemIndexStatusResponse,
)

indexes: Dict[str, ItemSearchEngine] = {}
store_dir = "/app/data/"


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
    if request.index_name in indexes:
        return SuccessResponse(
            success=True,
            message=f"Item index '{request.index_name}' created successfully",
            index_name=request.index_name,
        )

        # raise HTTPException(
        #     status_code=409,
        #     detail=f"Index '{request.index_name}' already exists",
        # )

    try:
        params = request.params
        path = store_dir + request.index_name

        if params:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                M=params.M,
                ef_construction=params.ef_construction,
                ef_search=params.ef_search,
                seed=params.rng_seed,
                tokenizer_path=request.tokenizer_path,
            )
        else:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                tokenizer_path=request.tokenizer_path,
            )

        indexes[request.index_name] = engine

        return SuccessResponse(
            success=True,
            message=f"Item index '{request.index_name}' created successfully",
            index_name=request.index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to create item index: {str(e)}")


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
        raise HTTPException(status_code=400, detail=f"Failed to delete item index: {str(e)}")


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
                M=params.M,
                ef_construction=params.ef_construction,
                ef_search=params.ef_search,
                seed=params.rng_seed,
                tokenizer_path=request.tokenizer_path,
            )
        else:
            engine = ItemSearchEngine(
                path,
                dim=request.dim,
                tokenizer_path=request.tokenizer_path,
            )
        indexes[index_name] = engine

        return SuccessResponse(
            success=True,
            message=f"Item index '{index_name}' loaded successfully",
            index_name=index_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to load item index: {str(e)}")


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
        raise HTTPException(status_code=400, detail=f"Failed to initialize item index: {str(e)}")


@app.post("/ingest")
async def ingest_single(request: IngestItemRequest):
    engine = get_engine(request.index_name)

    try:
        item = request.item

        engine.ingest(
            item.id,
            title=item.title,
            category=item.category or "",
            subcategory=item.subcategory or "",
            attributes=item.attributes,
        )

        return Response(
            content=b'{"success":true}',
            media_type="application/json",
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to ingest item: {str(e)}")


@app.post("/ingest/batch")
async def ingest_batch(request: Request):
    """
    JSON batch endpoint.

    Expected body:
    {
      "index_name": "wands_item_bench",
      "items": [
        {
          "id": "1",
          "title": "...",
          "category": "...",
          "subcategory": "..."
        }
      ]
    }
    """
    try:
        body = await request.body()
        data = orjson.loads(body)

        index_name = data.get("index_name")
        items = data.get("items")

        if not index_name:
            raise HTTPException(status_code=400, detail="Missing required field: index_name")

        if not isinstance(items, list):
            raise HTTPException(status_code=400, detail="Missing required list field: items")

        engine = get_engine(index_name)

        count = 0

        for item in items:
            external_id = str(item.get("id", ""))
            title = item.get("title", "")
            category = item.get("category", "") or ""
            subcategory = item.get("subcategory", "") or ""
            attributes = item.get("attributes", None)

            if not external_id:
                raise HTTPException(status_code=400, detail="Item missing required field: id")

            if not title:
                raise HTTPException(status_code=400, detail=f"Item '{external_id}' missing required field: title")

            engine.ingest(
                external_id,
                title=title,
                category=category,
                subcategory=subcategory,
                attributes=attributes,
            )

            count += 1

        return Response(
            content=orjson.dumps({"success": True, "count": count}),
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to ingest item batch: {str(e)}")


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
        raise HTTPException(status_code=400, detail=f"Failed to finalize item index: {str(e)}")


@app.post("/search")
async def search_items(request: SearchItemRequest):
    engine = get_engine(request.index_name)

    try:
        title = request.title or request.query

        if not title:
            raise HTTPException(
                status_code=400,
                detail="Missing query text. Provide either 'query' or 'title'.",
            )

        kwargs = {
            "query": title,
            "category": request.category or "",
            "subcategory": request.subcategory or "",
            "attributes": request.attributes,
            "k": request.k,
        }

        if request.ef_search is not None:
            kwargs["efs"] = request.ef_search

        ids = engine.search(**kwargs)
        return ids

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to search item index: {str(e)}")


@app.post("/search_with_distance")
async def search_items_with_distance(request: SearchItemRequest):
    engine = get_engine(request.index_name)

    try:
        title = request.title or request.query

        if not title:
            raise HTTPException(
                status_code=400,
                detail="Missing query text. Provide either 'query' or 'title'.",
            )

        kwargs = {
            "title": title,
            "category": request.category or "",
            "subcategory": request.subcategory or "",
            "attributes": request.attributes or [],
            "k": request.k,
        }

        if request.ef_search is not None:
            kwargs["ef_search"] = request.ef_search

        raw_results = engine.search_with_distance(**kwargs)
        return raw_results
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to search item index: {str(e)}")


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
                seed=params.rng_seed,
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
        raise HTTPException(status_code=400, detail=f"Failed to rebuild item index: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1984)
