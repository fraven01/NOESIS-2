"""Stub task that would remove PII from text."""

from __future__ import annotations

import json
import logging
from typing import Dict

from pydantic import BaseModel

from ..celery_app import app
from ...infra import pii as pii_util

logger = logging.getLogger(__name__)


class PiiMaskInput(BaseModel):
    text: str


class PiiMaskOutput(BaseModel):
    text: str


@app.task(name="pii_mask", queue="pii_mask")
def pii_mask(payload: Dict) -> Dict:
    inp = PiiMaskInput(**payload)
    logger.info(pii_util.mask(json.dumps({"task": "pii_mask", "input": inp.dict()})))
    out = PiiMaskOutput(text=pii_util.mask(inp.text))
    logger.info(pii_util.mask(json.dumps({"task": "pii_mask", "output": out.dict()})))
    return out.dict()
