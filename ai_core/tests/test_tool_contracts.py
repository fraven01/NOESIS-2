"""Golden sample tests for the tool contract envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
from json import loads as json_loads

import pytest
from pydantic import BaseModel, TypeAdapter

from ai_core.tool_contracts.base import (
    ToolContext,
    ToolError,
    ToolErrorDetail,
    ToolErrorMeta,
    ToolOutput,
    ToolResult,
    ToolResultMeta,
)
from ai_core.tools.errors import ToolErrorType


class _TestInput(BaseModel):
    query: str


class _TestOutput(BaseModel):
    result: str


def test_tool_result_roundtrip() -> None:
    context = ToolContext(
        tenant_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        trace_id="trace-1",
        invocation_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        now_iso=datetime(2024, 5, 4, 12, 30, tzinfo=timezone.utc),
    )
    tool_input = _TestInput(query="hello")
    tool_output = _TestOutput(result="world")
    result = ToolResult[_TestInput, _TestOutput](
        input=tool_input,
        data=tool_output,
        meta=ToolResultMeta(took_ms=42, routing={"trace_id": context.trace_id}),
    )

    json_payload = result.model_dump_json()
    parsed = TypeAdapter(ToolOutput[_TestInput, _TestOutput]).validate_json(
        json_payload
    )

    assert parsed.status == "ok"
    assert isinstance(parsed, ToolResult)
    assert parsed.model_dump() == result.model_dump()


def test_tool_error_validation_roundtrip() -> None:
    tool_input = _TestInput(query="bad")
    error = ToolError[_TestInput](
        input=tool_input,
        error=ToolErrorDetail(
            type=ToolErrorType.VALIDATION,
            message="missing query",
        ),
        meta=ToolErrorMeta(took_ms=0),
    )

    json_payload = error.model_dump_json()
    parsed = TypeAdapter(ToolOutput[_TestInput, _TestOutput]).validate_json(
        json_payload
    )

    assert parsed.status == "error"
    assert isinstance(parsed, ToolError)
    assert parsed.error.type is ToolErrorType.VALIDATION
    assert parsed.meta.took_ms >= 0


def test_tool_error_upstream_roundtrip() -> None:
    tool_input = _TestInput(query="external")
    error = ToolError[_TestInput](
        input=tool_input,
        error=ToolErrorDetail(
            type=ToolErrorType.UPSTREAM,
            message="service unavailable",
            upstream_status=503,
        ),
        meta=ToolErrorMeta(took_ms=1200),
    )

    json_payload = error.model_dump_json()
    parsed_json = json_loads(json_payload)

    assert parsed_json["error"]["type"] == "UPSTREAM"
    assert parsed_json["error"]["upstream_status"] == 503

    parsed = TypeAdapter(ToolOutput[_TestInput, _TestOutput]).validate_json(
        json_payload
    )

    assert parsed.status == "error"
    assert parsed.error.upstream_status == 503


def test_tool_context_datetime_strictness() -> None:
    with pytest.raises(ValueError):
        ToolContext(
            tenant_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            trace_id="trace-1",
            invocation_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            now_iso=datetime(2024, 5, 4, 12, 30),
        )

    context_z = ToolContext(
        tenant_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        trace_id="trace-1",
        invocation_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        now_iso=datetime.fromisoformat("2024-05-04T12:30:00+00:00"),
    )

    assert context_z.now_iso.tzinfo is timezone.utc

    context_with_z = ToolContext(
        tenant_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        trace_id="trace-1",
        invocation_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        now_iso=datetime.strptime("2024-05-04T12:30:00Z", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        ),
    )

    assert context_with_z.now_iso.tzinfo is timezone.utc


def test_tool_error_type_serialization() -> None:
    detail = ToolErrorDetail(type=ToolErrorType.VALIDATION, message="oops")

    dumped = detail.model_dump()

    assert dumped["type"] == "VALIDATION"
