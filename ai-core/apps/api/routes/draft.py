"""Endpoint for generating drafts."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..resp_headers import apply_std_headers
from ...orchestrator.graphs import draft as draft_graph

router = APIRouter()


class DraftBody(BaseModel):
    type: str
    inputs: Optional[Any] = None


@router.post("/draft")
async def handle(body: DraftBody, request: Request):
    meta = request.state.meta
    result = draft_graph.run(body.type, body.inputs, meta)
    response = JSONResponse({"result": result})
    apply_std_headers(response, result["prompt_version"], meta["trace"])
    return response
