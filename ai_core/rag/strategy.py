from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai_core.infra.prompts import load
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.tool_contracts import ToolContext

logger = logging.getLogger(__name__)


DEFAULT_MAX_VARIANTS = 3
MAX_VARIANTS_CAP = 5


@dataclass(frozen=True)
class QueryVariantResult:
    queries: list[str]
    source: str
    prompt_version: str | None = None
    error: str | None = None


def _dedupe_queries(candidates: Sequence[str], *, max_items: int) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(candidate)
        if len(queries) >= max_items:
            break
    return queries


def _fallback_queries(query: str, *, max_items: int) -> list[str]:
    base = query.strip()
    candidates = [
        base,
        f"{base} overview",
        f"{base} key facts",
        f"{base} definition",
        f"{base} requirements",
    ]
    return _dedupe_queries(candidates, max_items=max_items)


def _extract_json_object(text: str) -> Mapping[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty response payload")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response payload missing JSON object")
    fragment = cleaned[start : end + 1]
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON payload") from exc
    if not isinstance(data, Mapping):
        raise ValueError("response payload must be a JSON object")
    return data


def _coerce_query_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("queries must be a sequence")
    return _dedupe_queries([str(item) for item in value], max_items=max_items)


def _scope_hint(context: ToolContext) -> str:
    business = context.business
    case_id = business.case_id or "none"
    collection_id = business.collection_id or "none"
    workflow_id = business.workflow_id or "none"
    return (
        f"case_id={case_id}; collection_id={collection_id}; workflow_id={workflow_id}"
    )


def _resolve_max_variants(value: object) -> int:
    try:
        candidate = int(str(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_VARIANTS
    if candidate <= 0:
        return DEFAULT_MAX_VARIANTS
    return min(candidate, MAX_VARIANTS_CAP)


def generate_query_variants(
    query: str,
    context: ToolContext,
    *,
    max_variants: int | None = None,
) -> QueryVariantResult:
    base = (query or "").strip()
    if not base:
        return QueryVariantResult(
            queries=[],
            source="fallback",
            error="empty_query",
        )

    max_items = _resolve_max_variants(
        max_variants or os.getenv("RAG_QUERY_MAX_VARIANTS")
    )
    mode = str(os.getenv("RAG_QUERY_TRANSFORM_MODE", "heuristic")).strip().lower()
    if mode in {"off", "disabled", "false"}:
        return QueryVariantResult(queries=[base], source="disabled")

    if mode != "llm":
        return QueryVariantResult(
            queries=_fallback_queries(base, max_items=max_items),
            source="heuristic",
        )

    prompt = load("retriever/query_transform")
    prompt_text = f"{prompt['text']}\n\nQuestion: {base}\nScope: {_scope_hint(context)}"
    metadata = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
        "user_id": context.scope.user_id,
        "prompt_version": prompt["version"],
    }
    try:
        response = llm_client.call("analyze", prompt_text, metadata)
        payload = _extract_json_object(str(response.get("text") or ""))
        queries = _coerce_query_list(payload.get("queries"), max_items=max_items)
    except (LlmClientError, RateLimitError, ValueError) as exc:
        logger.warning(
            "rag.query_transform.failed",
            extra={
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        return QueryVariantResult(
            queries=_fallback_queries(base, max_items=max_items),
            source="fallback",
            prompt_version=prompt.get("version"),
            error=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "rag.query_transform.failed",
            extra={
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        return QueryVariantResult(
            queries=_fallback_queries(base, max_items=max_items),
            source="fallback",
            prompt_version=prompt.get("version"),
            error=str(exc),
        )

    if base not in queries:
        queries = _dedupe_queries([base, *queries], max_items=max_items)
    return QueryVariantResult(
        queries=queries,
        source="llm",
        prompt_version=prompt.get("version"),
    )


def expand_query_variants(
    query: str,
    existing: Sequence[str],
    *,
    max_variants: int | None = None,
) -> list[str]:
    base = (query or "").strip()
    if not base:
        return []
    max_items = _resolve_max_variants(
        max_variants or os.getenv("RAG_QUERY_MAX_VARIANTS")
    )
    expansions = [
        base,
        f"{base} overview",
        f"{base} background",
        f"{base} general principles",
        f"{base} summary",
    ]
    return _dedupe_queries([*existing, *expansions], max_items=max_items)


__all__ = ["QueryVariantResult", "generate_query_variants", "expand_query_variants"]
