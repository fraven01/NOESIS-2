"""Endpoint for answering questions via the orchestrator graph."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..resp_headers import apply_std_headers
from ...orchestrator.graphs import answer as answer_graph

router = APIRouter()


class AnswerBody(BaseModel):
    question: str


@router.post("/answer")
async def handle(body: AnswerBody, request: Request):
    meta = request.state.meta
    result = answer_graph.run(body.question, meta)
    payload = {k: v for k, v in result.items() if k != "prompt_version"}
    response = JSONResponse({"result": payload})
    apply_std_headers(response, result["prompt_version"], meta["trace"])
    return response
