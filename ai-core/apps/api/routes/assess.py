"""Endpoint for performing assessments."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..resp_headers import apply_std_headers
from ...orchestrator.graphs import assess as assess_graph

router = APIRouter()


class AssessBody(BaseModel):
    items: Optional[List[str]] = None


@router.post("/assess")
async def handle(body: AssessBody, request: Request):
    meta = request.state.meta
    result = assess_graph.run(body.items, meta)
    response = JSONResponse({"result": result})
    apply_std_headers(response, result["prompt_version"], meta["trace"])
    return response
