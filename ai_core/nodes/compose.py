from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping
import json
import re
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


def run_extract_questions(
    context: ToolContext,
    params: ComposeInput,
) -> ComposeOutput:
    """Extract question-style prompts from retrieved snippets using the LLM."""
    prompt = load("retriever/extract_questions")
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


def _parse_tag(text: str, tag: str) -> str:
    """Extract content between XML-like tags."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_rag_response(
    raw_text: str,
    *,
    pii_config: Mapping[str, Any],
) -> RagResponse:
    # Try Tag-based parsing first
    thought = _parse_tag(raw_text, "thought")
    answer = _parse_tag(raw_text, "answer")
    meta_json = _parse_tag(raw_text, "meta")

    if thought or answer or meta_json:
        # Structured Tag Format
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except json.JSONDecodeError:
            meta = {}

        # Parse used_sources with individual error handling
        used_sources: list[SourceRef] = []
        for s in meta.get("used_sources", []):
            if not isinstance(s, dict):
                continue
            try:
                used_sources.append(SourceRef.model_validate(s))
            except ValidationError:
                # Skip malformed source entries but continue parsing
                pass

        response = RagResponse(
            reasoning=RagReasoning(analysis=thought, gaps=[]),
            answer_markdown=answer
            or raw_text,  # Fallback to full text if <answer> missing
            used_sources=used_sources,
            suggested_followups=meta.get("suggested_followups", []),
        )
    else:
        # Fallback to legacy JSON format if tags not found (for transitioning)
        try:
            payload = json.loads(raw_text)
            response = RagResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            # Absolute fallback: treat as raw answer string
            response = RagResponse(
                reasoning=RagReasoning(
                    analysis="Parsing failed, showing raw output.", gaps=[]
                ),
                answer_markdown=raw_text,
                used_sources=[],
                suggested_followups=[],
            )

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
        # We no longer enforce "json_object" strictly in the LLM param
        # because we use Tags, but we can still request it if we want
        # the model to be more structured overall.
        # For tags, we just call it normally.
        result = client.call(
            "synthesize",
            masked,
            metadata,
        )
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

    text_payload = result.get("text") or ""
    debug_meta = _build_debug_meta(result)
    rag_response = _parse_rag_response(text_payload, pii_config=pii_config)

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

    # We use a primitive tag-aware stream proxy.
    # For now, we still buffer the full answer to parse logic,
    # but we stream the <answer> part to the user immediately.

    in_answer_tag = False

    try:
        for chunk in client.call_stream("synthesize", masked_prompt, metadata):
            if chunk.get("event") == "delta":
                text = chunk.get("text") or ""
                if not text:
                    continue
                answer_parts.append(text)

                # Basic stream interceptor for <answer> tag
                full_so_far = "".join(answer_parts)
                if "<answer>" in full_so_far and not in_answer_tag:
                    in_answer_tag = True
                    # If the tag just opened, we might have part of the answer already
                    # but usually it splits between chunks.
                    pass

                if in_answer_tag:
                    if "</answer>" in text:
                        # Close tag reached
                        part_before = text.split("</answer>")[0]
                        if part_before:
                            stream_callback(part_before)
                        in_answer_tag = False
                    else:
                        # Strip opening tag if it's in this chunk
                        out_text = text.replace("<answer>", "")
                        if out_text:
                            stream_callback(out_text)
                elif not any(
                    tag in full_so_far for tag in ["<thought>", "<meta>", "<answer>"]
                ):
                    # If NO tags are present yet, assume legacy streaming behavior
                    # until a tag is detected
                    stream_callback(text)

            elif chunk.get("event") == "error":
                raise LlmClientError(chunk.get("error") or "LLM error")
            elif chunk.get("event") == "final":
                final_meta = dict(chunk)
    except RateLimitError as exc:
        raise ToolRateLimitedError(str(getattr(exc, "detail", "rate limited"))) from exc
    except LlmClientError as exc:
        # ... (error handling same as before)
        raise exc

    raw_text = "".join(answer_parts)
    debug_meta = _build_debug_meta(final_meta)

    rag_response = _parse_rag_response(raw_text, pii_config=pii_config)

    return ComposeOutput(
        answer=rag_response.answer_markdown,
        prompt_version=prompt["version"],
        reasoning=rag_response.reasoning,
        used_sources=rag_response.used_sources,
        suggested_followups=rag_response.suggested_followups,
        debug_meta=debug_meta or None,
    )
