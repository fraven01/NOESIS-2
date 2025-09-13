"""Endpoint for preliminary checks."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..resp_headers import apply_std_headers
from ...orchestrator.graphs import precheck as precheck_graph

router = APIRouter()


class PrecheckBody(BaseModel):
    context: str


@router.post("/precheck")
async def handle(body: PrecheckBody, request: Request):
    meta = request.state.meta
    result = precheck_graph.run(body.context, meta)
    response = JSONResponse({"result": result})
    apply_std_headers(response, result["prompt_version"], meta["trace"])
    return response
