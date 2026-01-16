from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai_core.infra.prompts import load
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.tool_contracts import ToolContext


logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES = 6
DEFAULT_MAX_CHARS = 2000
MAX_MESSAGES_CAP = 12
MAX_CHARS_CAP = 4000


@dataclass(frozen=True)
class StandaloneQuestionResult:
    question: str
    source: str
    prompt_version: str | None = None
    error: str | None = None


def _resolve_int_env(name: str, fallback: int, *, cap: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return fallback
    if value <= 0:
        return fallback
    return min(value, cap)


def _normalize_role(value: object) -> str:
    if not isinstance(value, str):
        return "Other"
    lowered = value.strip().lower()
    if lowered in {"user", "human", "customer"}:
        return "User"
    if lowered in {"assistant", "ai", "agent"}:
        return "Assistant"
    return "Other"


def _format_history(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    max_messages: int,
    max_chars: int,
) -> str:
    if not history:
        return ""

    lines: list[str] = []
    total_chars = 0
    for entry in reversed(history):
        if len(lines) >= max_messages:
            break
        if not isinstance(entry, Mapping):
            continue
        role = _normalize_role(entry.get("role"))
        content = entry.get("content")
        if content is None:
            content = entry.get("text")
        content_text = str(content or "").strip()
        if not content_text:
            continue
        line = f"{role}: {content_text}"
        line_len = len(line)
        if max_chars and total_chars + line_len > max_chars:
            if not lines:
                lines.append(line[-max_chars:])
            break
        lines.append(line)
        total_chars += line_len

    if not lines:
        return ""
    return "\n".join(reversed(lines)).strip()


def _scope_hint(context: ToolContext) -> str:
    business = context.business
    case_id = business.case_id or "none"
    collection_id = business.collection_id or "none"
    workflow_id = business.workflow_id or "none"
    thread_id = business.thread_id or "none"
    return (
        "case_id="
        f"{case_id}; collection_id={collection_id}; workflow_id={workflow_id}; "
        f"thread_id={thread_id}"
    )


def _clean_response(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    lowered = cleaned.lower()
    for prefix in ("standalone question:", "standalone:", "question:"):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned.strip(" \"'`")


def generate_standalone_question(
    question: str,
    history: Sequence[Mapping[str, Any]] | None,
    context: ToolContext,
) -> StandaloneQuestionResult:
    base = (question or "").strip()
    if not base:
        return StandaloneQuestionResult(
            question="", source="empty", error="empty_query"
        )

    max_messages = _resolve_int_env(
        "RAG_CHAT_HISTORY_MAX_MESSAGES", DEFAULT_MAX_MESSAGES, cap=MAX_MESSAGES_CAP
    )
    max_chars = _resolve_int_env(
        "RAG_CHAT_HISTORY_MAX_CHARS", DEFAULT_MAX_CHARS, cap=MAX_CHARS_CAP
    )
    history_text = _format_history(
        history,
        max_messages=max_messages,
        max_chars=max_chars,
    )
    if not history_text:
        return StandaloneQuestionResult(question=base, source="original")

    prompt = load("retriever/standalone_question")
    prompt_text = (
        f"{prompt['text']}\n\nConversation:\n{history_text}\n\n"
        f"Follow-up question: {base}\nScope: {_scope_hint(context)}"
    )
    metadata = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
        "user_id": context.scope.user_id,
        "prompt_version": prompt["version"],
    }

    try:
        response = llm_client.call("analyze", prompt_text, metadata)
        standalone = _clean_response(str(response.get("text") or ""))
        if not standalone:
            raise ValueError("empty response")
    except (LlmClientError, RateLimitError, ValueError) as exc:
        logger.warning(
            "rag.standalone_question.failed",
            extra={"error": type(exc).__name__, "error_message": str(exc)},
        )
        return StandaloneQuestionResult(
            question=base,
            source="fallback",
            prompt_version=prompt.get("version"),
            error=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "rag.standalone_question.failed",
            extra={"error": type(exc).__name__, "error_message": str(exc)},
        )
        return StandaloneQuestionResult(
            question=base,
            source="fallback",
            prompt_version=prompt.get("version"),
            error=str(exc),
        )

    return StandaloneQuestionResult(
        question=standalone,
        source="llm",
        prompt_version=prompt.get("version"),
    )


__all__ = ["StandaloneQuestionResult", "generate_standalone_question"]
