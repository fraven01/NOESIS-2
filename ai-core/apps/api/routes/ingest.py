"""Endpoint for ingesting raw data references."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..resp_headers import apply_std_headers
from ...workers.tasks.ingest_raw import ingest_raw

router = APIRouter()


class IngestBody(BaseModel):
    upload_ref: str


@router.post("/ingest")
async def handle(body: IngestBody, request: Request):
    """Queue the ingest_raw task for the given reference."""
    ingest_raw.delay({"data": body.upload_ref})
    response = JSONResponse({"status": "queued"})
    apply_std_headers(response, "v1", request.state.meta["trace"])
    return response
