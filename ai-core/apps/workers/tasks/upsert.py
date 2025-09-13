"""Stub task for upserting embeddings into a vector store."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from pydantic import BaseModel

from ..celery_app import app
from ...infra import pii

logger = logging.getLogger(__name__)


class UpsertInput(BaseModel):
    embeddings: List[List[float]]


class UpsertOutput(BaseModel):
    status: str


@app.task(name="upsert", queue="upsert")
def upsert(payload: Dict) -> Dict:
    inp = UpsertInput(**payload)
    logger.info(pii.mask(json.dumps({"task": "upsert", "input": inp.dict()})))
    out = UpsertOutput(status="ok")
    logger.info(pii.mask(json.dumps({"task": "upsert", "output": out.dict()})))
    return out.dict()
