"""Tests for tool contracts and error taxonomy."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.tool_contracts import (
    InputError,
    InternalToolError,
    NotFoundError,
    RateLimitedError,
    TimeoutError,
    ToolContext,
    ToolError,
    UpstreamServiceError,
)


class TestToolContext:
    def test_valid_context_defaults(self) -> None:
        context = ToolContext(tenant_id="tenant", case_id="case")

        assert context.tenant_id == "tenant"
        assert context.case_id == "case"
        assert context.trace_id is None
        assert context.idempotency_key is None

    def test_missing_required_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ToolContext(tenant_id="tenant")  # type: ignore[call-arg]


class TestToolErrorHierarchy:
    @pytest.mark.parametrize(
        "error_cls",
        [
            InputError,
            NotFoundError,
            RateLimitedError,
            TimeoutError,
            UpstreamServiceError,
            InternalToolError,
        ],
    )
    def test_specialized_errors_inherit_from_tool_error(self, error_cls: type[ToolError]) -> None:
        assert issubclass(error_cls, ToolError)

    def test_tool_error_inherits_from_exception(self) -> None:
        assert issubclass(ToolError, Exception)
