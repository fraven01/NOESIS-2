"""Endpoint for proposing solutions."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..resp_headers import apply_std_headers
from ...orchestrator.graphs import solve as solve_graph

router = APIRouter()


class SolveBody(BaseModel):
    issue: str


@router.post("/solve")
async def handle(body: SolveBody, request: Request):
    meta = request.state.meta
    result = solve_graph.run(body.issue, meta)
    response = JSONResponse({"result": result})
    apply_std_headers(response, result["prompt_version"], meta["trace"])
    return response
