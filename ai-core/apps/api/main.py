"""FastAPI application exposing AI Core MVP endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .middleware.guards import assert_case_active
from .routes import answer, assess, draft, ingest, precheck, solve
from ..infra import object_store, rate_limit

app = FastAPI()
app.middleware("http")(assert_case_active)

app.include_router(ingest.router)
app.include_router(answer.router)
app.include_router(assess.router)
app.include_router(draft.router)
app.include_router(solve.router)
app.include_router(precheck.router)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe verifying Redis and filesystem access."""
    redis_ok = rate_limit.ready()
    fs_ok = object_store.ready()
    status_code = 200 if redis_ok and fs_ok else 503
    return JSONResponse(
        status_code=status_code, content={"redis": redis_ok, "fs": fs_ok}
    )
