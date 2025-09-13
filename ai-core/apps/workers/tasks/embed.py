"""Stub task for embedding chunks."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from pydantic import BaseModel

from ..celery_app import app
from ...infra import pii

logger = logging.getLogger(__name__)


class EmbedInput(BaseModel):
    chunks: List[str]


class EmbedOutput(BaseModel):
    embeddings: List[List[float]]


@app.task(name="embed", queue="embed")
def embed(payload: Dict) -> Dict:
    inp = EmbedInput(**payload)
    logger.info(pii.mask(json.dumps({"task": "embed", "input": inp.dict()})))
    out = EmbedOutput(embeddings=[[0.0] for _ in inp.chunks])
    logger.info(pii.mask(json.dumps({"task": "embed", "output": out.dict()})))
    return out.dict()
