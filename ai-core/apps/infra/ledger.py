"""Evidence ledger stub that logs JSON entries."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from . import pii

logger = logging.getLogger(__name__)


def record(meta: Dict[str, Any]) -> None:
    """Log a metadata dictionary as a masked JSON string."""
    logger.info(pii.mask(json.dumps(meta)))
