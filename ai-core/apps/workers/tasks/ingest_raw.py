"""Stub task for ingesting raw references."""

from __future__ import annotations

import json
import logging
from typing import Dict

from celery import chain
from pydantic import BaseModel

from ..celery_app import app
from .extract_text import extract_text
from .pii_mask import pii_mask
from .chunk import chunk
from .embed import embed
from .upsert import upsert
from ...infra import pii

logger = logging.getLogger(__name__)


class IngestRawInput(BaseModel):
    data: str


class IngestRawOutput(BaseModel):
    queued: bool


@app.task(name="ingest_raw", queue="ingest_raw")
def ingest_raw(payload: Dict) -> Dict:
    """Trigger the processing chain for the given raw data."""
    inp = IngestRawInput(**payload)
    logger.info(pii.mask(json.dumps({"task": "ingest_raw", "input": inp.dict()})))
    chain(
        extract_text.s({"data": inp.data}),
        pii_mask.s(),
        chunk.s(),
        embed.s(),
        upsert.s(),
    ).delay()
    out = IngestRawOutput(queued=True)
    logger.info(pii.mask(json.dumps({"task": "ingest_raw", "output": out.dict()})))
    return out.dict()
