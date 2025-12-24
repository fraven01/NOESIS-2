from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from ai_core.tool_contracts.base import ToolContext


@pytest.fixture
def base_payload() -> dict:
    return {
        "tenant_id": uuid4(),
        "trace_id": "trace-1",
        "invocation_id": uuid4(),
        "now_iso": datetime.now(timezone.utc),
    }


def test_tool_context_both_run_ids_allowed(base_payload: dict) -> None:
    """Both run_id and ingestion_run_id can co-exist (Pre-MVP ID Contract)."""
    context = ToolContext(
        **base_payload, run_id="run-1", ingestion_run_id="ingestion-run-1"
    )
    assert context.run_id == "run-1"
    assert context.ingestion_run_id == "ingestion-run-1"


def test_tool_context_no_run_ids_fails(base_payload: dict) -> None:
    with pytest.raises(ValueError):
        ToolContext(**base_payload)


def test_tool_context_with_run_id_passes(base_payload: dict) -> None:
    context = ToolContext(**base_payload, run_id="run-1")
    assert context.run_id == "run-1"
    assert context.ingestion_run_id is None


def test_tool_context_with_ingestion_run_id_passes(base_payload: dict) -> None:
    context = ToolContext(**base_payload, ingestion_run_id="ingestion-run-1")
    assert context.ingestion_run_id == "ingestion-run-1"
    assert context.run_id is None
