from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def record(meta: dict[str, Any]) -> None:
    """Log metadata via structured logger output."""
    extra = {
        "tenant_id": meta.get("tenant") or meta.get("tenant_id"),
        "trace_id": meta.get("trace_id"),
        "invocation_id": meta.get("invocation_id"),
        "ledger": meta,
    }
    LOGGER.info("ledger.record", extra=extra)
