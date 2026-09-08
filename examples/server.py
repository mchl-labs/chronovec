"""Minimal FastAPI HTTP/REST wrapper around a ChronoVec index.

Exposes insert, delete, search, vacuum, and snapshot endpoints so
frameworks that prefer a network service (LangChain remote retriever,
any HTTP client) can talk to a ChronoVec index without embedding the
library directly.

Install:
    pip install ".[dev]" fastapi uvicorn[standard]

Run:
    uvicorn examples.server:app --reload
    # or from the project root:
    python -m uvicorn examples.server:app --host 0.0.0.0 --port 8000

The server is single-process. ChronoVec searches are lock-free, so many
concurrent reads are fine; writes serialize internally on the MVCC lock
(consistent with the library's single-writer design).

For production use, wrap the uvicorn process in a process supervisor,
add authentication middleware, and optionally front it with gunicorn using
the uvicorn worker class.
"""

from __future__ import annotations

import importlib.util

# FastAPI is an optional dependency; check before importing.
_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

print(
    "ChronoVec server. Start with: uvicorn examples.server:app --reload"
    if _FASTAPI_AVAILABLE
    else "server.py requires fastapi: pip install fastapi uvicorn[standard]"
)

if _FASTAPI_AVAILABLE:
    import asyncio
    from typing import Annotated

    import numpy as np
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel, Field

    from chronovec import Index

    # ---------------------------------------------------------------------------
    # Global index; adjust dimensions and metric at startup.
    # ---------------------------------------------------------------------------
    DIMENSIONS = 384
    METRIC = "cosine"

    _index = Index(DIMENSIONS, metric=METRIC, nprobe=16)

    app = FastAPI(
        title="ChronoVec",
        description="Snapshot-isolated vector index over HTTP.",
        version="0.1.0",
    )

    # -------------------------------------------------------------------------
    # Request / response models
    # -------------------------------------------------------------------------

    class InsertRequest(BaseModel):
        id: int
        vector: list[float] = Field(..., min_length=1)

    class DeleteRequest(BaseModel):
        id: int

    class SearchRequest(BaseModel):
        vector: list[float] = Field(..., min_length=1)
        k: int = Field(10, ge=1, le=1000)
        nprobe: int | None = Field(None, ge=1)
        snapshot: int | None = None

    class SearchHit(BaseModel):
        id: int
        distance: float

    class SearchResponse(BaseModel):
        hits: list[SearchHit]
        snapshot: int

    class VacuumRequest(BaseModel):
        oldest_snapshot: int | None = None
        budget_versions: int = Field(256, ge=0)

    class VacuumResponse(BaseModel):
        reclaimed: int
        clock: int

    class StatsResponse(BaseModel):
        clock: int
        dimensions: int
        metric: str

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _vec(raw: list[float]) -> np.ndarray:
        v = np.array(raw, dtype=np.float32)
        if v.shape != (DIMENSIONS,):
            raise HTTPException(
                status_code=422,
                detail=f"vector must have {DIMENSIONS} dimensions, got {len(raw)}",
            )
        return v

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    @app.get("/", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        return StatsResponse(clock=_index.clock, dimensions=DIMENSIONS, metric=METRIC)

    @app.post("/insert", status_code=201)
    async def insert(req: InsertRequest) -> dict:
        v = _vec(req.vector)
        committed = await asyncio.to_thread(_index.insert, req.id, v)
        return {"id": req.id, "committed": committed}

    @app.post("/delete")
    async def delete(req: DeleteRequest) -> dict:
        committed = await asyncio.to_thread(_index.delete, req.id)
        return {"id": req.id, "committed": committed}

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        v = _vec(req.vector)
        kwargs: dict = {"k": req.k}
        if req.nprobe is not None:
            kwargs["nprobe"] = req.nprobe
        if req.snapshot is not None:
            kwargs["snapshot"] = req.snapshot
        hits = await asyncio.to_thread(_index.search, v, **kwargs)
        snap = req.snapshot if req.snapshot is not None else _index.clock
        return SearchResponse(
            hits=[SearchHit(id=h.id, distance=h.distance) for h in hits],
            snapshot=snap,
        )

    @app.post("/vacuum", response_model=VacuumResponse)
    async def vacuum(req: VacuumRequest) -> VacuumResponse:
        oldest = req.oldest_snapshot if req.oldest_snapshot is not None else _index.clock + 1
        reclaimed = await asyncio.to_thread(
            _index.vacuum, oldest_snapshot=oldest, budget_versions=req.budget_versions
        )
        return VacuumResponse(reclaimed=reclaimed, clock=_index.clock)

    @app.post("/save")
    async def save(path: Annotated[str, Body(embed=True)] = "index.cvec") -> dict:
        await asyncio.to_thread(_index.save, path)
        return {"saved": path}

    @app.post("/load")
    async def load(path: Annotated[str, Body(embed=True)] = "index.cvec") -> dict:
        global _index
        _index = await asyncio.to_thread(Index.load, path)  # type: ignore[attr-defined]
        return {"loaded": path, "clock": _index.clock}
