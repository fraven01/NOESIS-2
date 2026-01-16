from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.observability import observe_span
from ai_core.llm import client
from ai_core.tool_contracts import ToolContext
from ai_core.tool_contracts import (
    RateLimitedError as ToolRateLimitedError,
    TimeoutError as ToolTimeoutError,
    UpstreamServiceError,
)
from ai_core.llm.client import LlmClientError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field


class ComposeInput(BaseModel):
    """Structured input parameters for the compose node."""

    question: str | None = None
    snippets: list[Mapping[str, Any]] = Field(default_factory=list)
    stream_callback: Callable[[str], None] | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


class ComposeOutput(BaseModel):
    """Structured output payload returned by the compose node."""

    answer: str | None
    prompt_version: str | None
    snippets: list[Mapping[str, Any]] | None = None
    retrieval: Mapping[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


def _build_llm_metadata(
    context: ToolContext,
    *,
    prompt_version: str,
) -> dict[str, Any]:
    metadata = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
        "user_id": context.scope.user_id,
        "prompt_version": prompt_version,
    }
    key_alias = context.metadata.get("key_alias")
    if key_alias:
        metadata["key_alias"] = key_alias
    ledger_logger = context.metadata.get("ledger_logger")
    if ledger_logger:
        metadata["ledger_logger"] = ledger_logger
    return metadata


def run(
    context: ToolContext,
    params: ComposeInput,
) -> ComposeOutput:
    """Compose an answer from retrieved snippets using the LLM."""
    prompt = load("retriever/answer")
    metadata = _build_llm_metadata(context, prompt_version=prompt["version"])
    return _run(
        prompt,
        params,
        metadata=metadata,
        stream_callback=params.stream_callback,
    )


def _resolve_snippet_label(snippet: Mapping[str, Any], index: int) -> str:
    raw_label = snippet.get("citation")
    if not isinstance(raw_label, str) or not raw_label.strip():
        for candidate in (snippet.get("source"), snippet.get("id")):
            if isinstance(candidate, str) and candidate.strip():
                raw_label = candidate
                break
        else:
            raw_label = f"Snippet {index + 1}"
    return raw_label.strip()


def _format_snippet_context(snippets: Iterable[Mapping[str, Any]]) -> str:
    formatted: list[str] = []
    for index, snippet in enumerate(snippets):
        text_value = snippet.get("text")
        text = str(text_value or "").strip()
        label = _resolve_snippet_label(snippet, index)
        if text:
            formatted.append(f"[{label}] {text}")
        else:
            formatted.append(f"[{label}]")
    return "\n".join(formatted)


@observe_span(name="compose")
def _run(
    prompt: Dict[str, str],
    params: ComposeInput,
    *,
    metadata: Dict[str, Any],
    stream_callback: Callable[[str], None] | None = None,
) -> ComposeOutput:
    snippets_data = [
        snippet for snippet in params.snippets if isinstance(snippet, Mapping)
    ]
    snippets_text = _format_snippet_context(snippets_data)
    question = params.question or ""
    full_prompt = f"{prompt['text']}\n\nQuestion: {question}\nContext:\n{snippets_text}"
    pii_config = get_pii_config()
    masked = mask_prompt(full_prompt, config=pii_config)
    if stream_callback is not None:
        return _run_stream(
            prompt,
            params,
            masked_prompt=masked,
            metadata=metadata,
            stream_callback=stream_callback,
        )

    try:
        result = client.call("synthesize", masked, metadata)
    except RateLimitError as exc:
        raise ToolRateLimitedError(str(getattr(exc, "detail", "rate limited"))) from exc
    except LlmClientError as exc:
        status = getattr(exc, "status", None)
        # Map LLM client errors to tool-level typed errors for the graph layer
        try:
            code = int(status) if isinstance(status, int) else int(str(status))
        except Exception:
            code = None
        message = str(getattr(exc, "detail", None) or exc) or "LLM error"
        if code in {408, 504}:
            raise ToolTimeoutError(message) from exc
        # For remaining 4xx/5xx, surface as upstream dependency error
        raise UpstreamServiceError(message) from exc
    answer = mask_response(result["text"], config=pii_config)
    return ComposeOutput(answer=answer, prompt_version=prompt["version"])


def _run_stream(
    prompt: Dict[str, str],
    params: ComposeInput,
    *,
    masked_prompt: str,
    metadata: Dict[str, Any],
    stream_callback: Callable[[str], None],
) -> ComposeOutput:
    pii_config = get_pii_config()
    answer_parts: list[str] = []
    try:
        for chunk in client.call_stream("synthesize", masked_prompt, metadata):
            if chunk.get("event") == "delta":
                text = chunk.get("text") or ""
                if text:
                    answer_parts.append(text)
                    stream_callback(text)
            elif chunk.get("event") == "error":
                raise LlmClientError(chunk.get("error") or "LLM error")
    except RateLimitError as exc:
        raise ToolRateLimitedError(str(getattr(exc, "detail", "rate limited"))) from exc
    except LlmClientError as exc:
        status = getattr(exc, "status", None)
        try:
            code = int(status) if isinstance(status, int) else int(str(status))
        except Exception:
            code = None
        message = str(getattr(exc, "detail", None) or exc) or "LLM error"
        if code in {408, 504}:
            raise ToolTimeoutError(message) from exc
        raise UpstreamServiceError(message) from exc

    answer = mask_response("".join(answer_parts), config=pii_config)
    return ComposeOutput(answer=answer, prompt_version=prompt["version"])
