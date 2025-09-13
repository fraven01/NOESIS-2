"""Stub task for extracting text from raw data."""

from __future__ import annotations

import json
import logging
from typing import Dict

from pydantic import BaseModel

from ..celery_app import app
from ...infra import pii

logger = logging.getLogger(__name__)


class ExtractTextInput(BaseModel):
    data: str


class ExtractTextOutput(BaseModel):
    text: str


@app.task(name="extract_text", queue="extract_text")
def extract_text(payload: Dict) -> Dict:
    inp = ExtractTextInput(**payload)
    logger.info(pii.mask(json.dumps({"task": "extract_text", "input": inp.dict()})))
    out = ExtractTextOutput(text=inp.data)
    logger.info(pii.mask(json.dumps({"task": "extract_text", "output": out.dict()})))
    return out.dict()
