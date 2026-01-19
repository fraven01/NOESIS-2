from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping
import json
from pathlib import Path

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.observability import observe_span
from ai_core.llm import client
from ai_core.rag.schemas import RagReasoning, RagResponse, SourceRef
from ai_core.tool_contracts import ToolContext
from ai_core.tool_contracts import (
    RateLimitedError as ToolRateLimitedError,
    TimeoutError as ToolTimeoutError,
    UpstreamServiceError,
)
from ai_core.llm.client import LlmClientError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
    reasoning: RagReasoning | None = None
    used_sources: list[SourceRef] | None = None
    suggested_followups: list[str] | None = None
    debug_meta: dict[str, Any] | None = None

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


def _build_prompt_text(
    prompt_text: str,
    *,
    question: str,
    snippets_text: str,
) -> str:
    return f"{prompt_text}\n\nQuestion: {question}\nContext:\n{snippets_text}"


def _load_prompt_version(alias: str, *, version: str) -> Dict[str, str]:
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    prompt_file = prompts_dir / f"{alias}.v{version}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt {alias}.v{version}.md not found")
    text = prompt_file.read_text(encoding="utf-8")
    return {"version": f"v{version}", "text": text}


def _mask_rag_response(
    response: RagResponse,
    *,
    pii_config: Mapping[str, Any],
) -> RagResponse:
    masked_analysis = mask_response(response.reasoning.analysis, config=pii_config)
    masked_gaps = [
        mask_response(gap, config=pii_config) if gap else gap
        for gap in response.reasoning.gaps
    ]
    masked_answer = mask_response(response.answer_markdown, config=pii_config)
    masked_followups = [
        mask_response(item, config=pii_config) if item else item
        for item in response.suggested_followups
    ]
    return response.model_copy(
        update={
            "reasoning": response.reasoning.model_copy(
                update={"analysis": masked_analysis, "gaps": masked_gaps}
            ),
            "answer_markdown": masked_answer,
            "suggested_followups": masked_followups,
        }
    )


def _build_debug_meta(result: Mapping[str, Any]) -> dict[str, Any]:
    usage = result.get("usage")
    cost_usd = result.get("cost_usd")
    if cost_usd is None and isinstance(usage, Mapping):
        cost = usage.get("cost")
        if isinstance(cost, Mapping):
            for key in ("usd", "USD", "total"):
                raw_value = cost.get(key)
                try:
                    cost_usd = float(raw_value)
                    break
                except (TypeError, ValueError):
                    continue
    return {
        "latency_ms": result.get("latency_ms"),
        "usage": usage,
        "model": result.get("model"),
        "cost_usd": cost_usd,
        "cost": result.get("cost"),
        "cache_hit": result.get("cache_hit"),
    }


def _parse_rag_response(
    raw_text: str,
    *,
    pii_config: Mapping[str, Any],
) -> RagResponse:
    payload = json.loads(raw_text)
    response = RagResponse.model_validate(payload)
    return _mask_rag_response(response, pii_config=pii_config)


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
    full_prompt = _build_prompt_text(
        prompt["text"],
        question=question,
        snippets_text=snippets_text,
    )
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
        result = client.call(
            "synthesize",
            masked,
            metadata,
            response_format={"type": "json_object"},
        )
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
    text_payload = result.get("text") or ""
    debug_meta = _build_debug_meta(result)
    try:
        rag_response = _parse_rag_response(text_payload, pii_config=pii_config)
    except (json.JSONDecodeError, ValidationError):
        fallback_prompt = _load_prompt_version("retriever/answer", version="1")
        fallback_meta = dict(metadata)
        fallback_meta["prompt_version"] = fallback_prompt["version"]
        fallback_full = _build_prompt_text(
            fallback_prompt["text"],
            question=question,
            snippets_text=snippets_text,
        )
        fallback_masked = mask_prompt(fallback_full, config=pii_config)
        try:
            fallback_result = client.call("synthesize", fallback_masked, fallback_meta)
        except RateLimitError as exc:
            raise ToolRateLimitedError(
                str(getattr(exc, "detail", "rate limited"))
            ) from exc
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
        answer = mask_response(fallback_result.get("text") or "", config=pii_config)
        return ComposeOutput(
            answer=answer,
            prompt_version=fallback_prompt["version"],
            debug_meta=_build_debug_meta(fallback_result),
        )

    return ComposeOutput(
        answer=rag_response.answer_markdown,
        prompt_version=prompt["version"],
        reasoning=rag_response.reasoning,
        used_sources=rag_response.used_sources,
        suggested_followups=rag_response.suggested_followups,
        debug_meta=debug_meta,
    )


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
    final_meta: dict[str, Any] = {}
    snippets_data = [
        snippet for snippet in params.snippets if isinstance(snippet, Mapping)
    ]
    snippets_text = _format_snippet_context(snippets_data)
    question = params.question or ""
    try:
        for chunk in client.call_stream("synthesize", masked_prompt, metadata):
            if chunk.get("event") == "delta":
                text = chunk.get("text") or ""
                if text:
                    answer_parts.append(text)
            elif chunk.get("event") == "error":
                raise LlmClientError(chunk.get("error") or "LLM error")
            elif chunk.get("event") == "final":
                final_meta = dict(chunk)
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

    raw_text = "".join(answer_parts)
    debug_meta = {
        "latency_ms": final_meta.get("latency_ms"),
        "usage": final_meta.get("usage"),
        "model": final_meta.get("model"),
        "cost_usd": final_meta.get("cost_usd"),
        "cost": final_meta.get("cost"),
        "cache_hit": final_meta.get("cache_hit"),
        "finish_reason": final_meta.get("finish_reason"),
    }
    try:
        rag_response = _parse_rag_response(raw_text, pii_config=pii_config)
    except (json.JSONDecodeError, ValidationError):
        fallback_prompt = _load_prompt_version("retriever/answer", version="1")
        fallback_meta = dict(metadata)
        fallback_meta["prompt_version"] = fallback_prompt["version"]
        fallback_full = _build_prompt_text(
            fallback_prompt["text"],
            question=question,
            snippets_text=snippets_text,
        )
        fallback_masked = mask_prompt(fallback_full, config=pii_config)
        try:
            fallback_result = client.call("synthesize", fallback_masked, fallback_meta)
        except RateLimitError as exc:
            raise ToolRateLimitedError(
                str(getattr(exc, "detail", "rate limited"))
            ) from exc
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
        answer = mask_response(fallback_result.get("text") or "", config=pii_config)
        if answer:
            stream_callback(answer)
        return ComposeOutput(
            answer=answer,
            prompt_version=fallback_prompt["version"],
            debug_meta=_build_debug_meta(fallback_result),
        )

    answer = rag_response.answer_markdown
    if answer:
        stream_callback(answer)
    return ComposeOutput(
        answer=answer,
        prompt_version=prompt["version"],
        reasoning=rag_response.reasoning,
        used_sources=rag_response.used_sources,
        suggested_followups=rag_response.suggested_followups,
        debug_meta=debug_meta or None,
    )
