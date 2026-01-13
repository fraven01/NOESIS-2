"""HITL payload helpers for collection search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_hitl_payload(
    *,
    ids: Mapping[str, Any],
    input_data: Mapping[str, Any],
    result_data: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "tenant_id": ids.get("tenant_id"),
        "question": input_data.get("question"),
        "top_k": result_data.get("top_k", []),
    }
