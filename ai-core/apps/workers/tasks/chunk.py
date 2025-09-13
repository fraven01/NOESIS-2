"""Stub task for splitting text into chunks."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from pydantic import BaseModel

from ..celery_app import app
from ...infra import pii

logger = logging.getLogger(__name__)


class ChunkInput(BaseModel):
    text: str


class ChunkOutput(BaseModel):
    chunks: List[str]


@app.task(name="chunk", queue="chunk")
def chunk(payload: Dict) -> Dict:
    inp = ChunkInput(**payload)
    logger.info(pii.mask(json.dumps({"task": "chunk", "input": inp.dict()})))
    out = ChunkOutput(chunks=[inp.text])
    logger.info(pii.mask(json.dumps({"task": "chunk", "output": out.dict()})))
    return out.dict()
