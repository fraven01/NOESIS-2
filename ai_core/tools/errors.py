"""Common error types for AI tool contracts."""

from __future__ import annotations

from typing import Any, Mapping

_DOC_HINT = "See README.md (Fehlercodes Abschnitt) for remediation guidance."


class ToolError(ValueError):
    """Base class for tool errors with structured metadata."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
        doc_hint: str | None = _DOC_HINT,
    ) -> None:
        detail = f"{code}: {message}"
        if doc_hint:
            detail = f"{detail}. {doc_hint}"
        super().__init__(detail)
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.doc_hint = doc_hint


class InputError(ToolError):
    """Raised when tool input validation fails."""


__all__ = ["ToolError", "InputError"]
