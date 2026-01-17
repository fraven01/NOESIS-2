"""Strategy helpers for collection search."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_core.infra.observability import update_observation
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from common.validators import normalise_str_sequence, optional_str, require_trimmed_str

LOGGER = logging.getLogger(__name__)

_MAX_POLICIES = 4
_MAX_SOURCES = 4
_MAX_ITEM_CHARS = 160
_MAX_NOTES_CHARS = 240


def _limit_items(values: Sequence[str], *, limit: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        if len(candidate) > _MAX_ITEM_CHARS:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(candidate)
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)


def _ensure_original_query(queries: list[str], original: str) -> list[str]:
    original_clean = original.strip()
    if not original_clean:
        return queries
    if not any(q.strip().casefold() == original_clean.casefold() for q in queries):
        queries.insert(0, original_clean)
    return queries[:5]


def _query_domains(query: str) -> set[str]:
    tokens = set()
    for match in re.findall(r"[a-z0-9-]+(?:\\.[a-z0-9-]+)+", query.casefold()):
        tokens.add(match)
    return tokens


def _filter_disallowed_sources(
    sources: Sequence[str], *, query: str
) -> tuple[str, ...]:
    query_domains = _query_domains(query)
    if not query_domains:
        return tuple(sources)
    filtered: list[str] = []
    for item in sources:
        item_key = item.casefold()
        if any(domain in item_key for domain in query_domains):
            continue
        filtered.append(item)
    return tuple(filtered)


class SearchStrategyRequest(BaseModel):
    """Normalised request payload for search strategy generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    query: str
    quality_mode: str
    purpose: str

    @field_validator("tenant_id", "query", "quality_mode", "purpose", mode="before")
    @classmethod
    def _trimmed(cls, value: Any) -> str:
        return require_trimmed_str(value)


class SearchStrategy(BaseModel):
    """Structured search strategy containing query expansions and policy hints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=7)
    policies_applied: tuple[str, ...] = Field(default_factory=tuple)
    preferred_sources: tuple[str, ...] = Field(default_factory=tuple)
    disallowed_sources: tuple[str, ...] = Field(default_factory=tuple)
    notes: str | None = None

    @field_validator("queries", mode="before")
    @classmethod
    def _normalise_queries(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("queries must be a sequence")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("query expansions must be strings")
            candidate = item.strip()
            if not candidate:
                continue
            cleaned.append(candidate)
        if not cleaned:
            raise ValueError("at least one query expansion must be provided")
        if len(cleaned) > 7:
            raise ValueError("no more than seven query expansions are allowed")
        return cleaned

    @field_validator(
        "policies_applied", "preferred_sources", "disallowed_sources", mode="before"
    )
    @classmethod
    def _normalise_sequences(cls, value: Any) -> tuple[str, ...]:
        return normalise_str_sequence(value)

    @field_validator("notes", mode="before")
    @classmethod
    def _normalise_notes(cls, value: Any) -> str | None:
        return optional_str(value, field_name="notes")


def fallback_strategy(request: SearchStrategyRequest) -> SearchStrategy:
    """Return a deterministic baseline strategy when LLM generation fails."""
    base_query = request.query
    purpose_hint = request.purpose.replace("_", " ")
    candidates = [
        base_query,
        f"{base_query} {purpose_hint}",
        f"{base_query} overview",
        f"{base_query} information",
        f"{base_query} guide",
    ]
    seen: set[str] = set()
    queries: list[str] = []
    for item in candidates:
        normalised = item.strip()
        if not normalised:
            continue
        if normalised.lower() in seen:
            continue
        seen.add(normalised.lower())
        queries.append(normalised)
        if len(queries) == 3:
            break
    return SearchStrategy(
        queries=queries,
        policies_applied=("default",),
        preferred_sources=(),
        disallowed_sources=(),
    )


def fallback_with_reason(
    request: SearchStrategyRequest, message: str, error: Exception | None = None
) -> SearchStrategy:
    if error is not None:
        LOGGER.warning(message, exc_info=error)
    else:
        LOGGER.warning(message)
    return fallback_strategy(request)


def coerce_query_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence):
        raise ValueError("queries must be a sequence")
    queries: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item in (None, ""):
            continue
        candidate = str(item).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(candidate)
        if len(queries) == 5:
            break
    if len(queries) < 3:
        raise ValueError("at least three queries required")
    return queries


def extract_strategy_payload(text: str) -> Mapping[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty strategy payload")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1])
            cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("payload does not contain a JSON object")
    fragment = cleaned[start : end + 1]
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON payload") from exc
    if not isinstance(data, Mapping):
        raise ValueError("strategy payload must be an object")
    return data


def llm_strategy_generator(request: SearchStrategyRequest) -> SearchStrategy:
    """Generate a search strategy via the production LLM client."""

    prompt = (
        "You are an expert research strategist tasked with "
        "designing focused web search strategies.\n"
        "Analyse the user's intent and produce between 3 and 5 focused web "
        "search queries that maximise authoritative and relevant sources.\n"
        "Always include the original query as one of the queries.\n"
        "Keep query expansions close to the original intent; avoid broad "
        "definitions or unrelated context.\n"
        "Only include preferred or disallowed sources when the user explicitly "
        "mentions them; otherwise keep those arrays empty.\n"
        "Respond with a JSON object containing the keys 'queries', "
        "'policies_applied', 'preferred_sources', 'disallowed_sources', and "
        "an optional 'notes'.\n"
        "- 'queries' must be an array of 3-5 strings.\n"
        "- Optional arrays may be empty if not applicable.\n"
        "Do not include any additional text outside the JSON object.\n"
        "\n"
        "Context:\n"
        f"- Tenant: {request.tenant_id}\n"
        f"- Purpose: {request.purpose}\n"
        f"- Quality mode: {request.quality_mode}\n"
        f"- Original query: {request.query}"
    )
    query_hash = str(uuid4())[:12]  # Simplified hash logic
    metadata = {
        "tenant_id": request.tenant_id,
        "case_id": f"collection-search:{request.purpose}:{query_hash}",
        "trace_id": None,
        "prompt_version": "collection_search_strategy_v1",
    }
    try:
        llm_start = time.time()
        response = llm_client.call("analyze", prompt, metadata)
        llm_latency = time.time() - llm_start

        # Track LLM metrics (simplified for brevity)
        update_observation(
            metadata={
                "llm.latency_ms": int(llm_latency * 1000),
                "llm.label": "analyze",
            }
        )
    except (LlmClientError, RateLimitError) as exc:
        return fallback_with_reason(
            request,
            "llm strategy generation failed; using fallback strategy",
            exc,
        )
    except Exception as exc:
        return fallback_with_reason(
            request,
            "llm strategy generation failed; using fallback strategy",
            exc,
        )

    text = (response.get("text") or "").strip()
    try:
        payload = extract_strategy_payload(text)
    except ValueError as exc:
        return fallback_with_reason(
            request,
            "unable to parse LLM strategy payload; using fallback strategy",
            exc,
        )

    try:
        queries = coerce_query_list(payload.get("queries"))
        queries = _ensure_original_query(queries, request.query)
    except Exception as exc:
        return fallback_with_reason(
            request,
            "invalid queries in LLM strategy payload; using fallback strategy",
            exc,
        )

    policies_raw = normalise_str_sequence(payload.get("policies_applied"))
    preferred_raw = normalise_str_sequence(payload.get("preferred_sources"))
    disallowed_raw = normalise_str_sequence(payload.get("disallowed_sources"))
    notes = payload.get("notes") if isinstance(payload.get("notes"), str) else None
    if isinstance(notes, str):
        notes = notes.strip() or None
        if notes and len(notes) > _MAX_NOTES_CHARS:
            notes = notes[:_MAX_NOTES_CHARS].rstrip()

    policies = _limit_items(policies_raw, limit=_MAX_POLICIES)
    preferred_sources = _limit_items(preferred_raw, limit=_MAX_SOURCES)
    disallowed_sources = _limit_items(disallowed_raw, limit=_MAX_SOURCES)
    disallowed_sources = _filter_disallowed_sources(
        disallowed_sources, query=request.query
    )

    try:
        return SearchStrategy(
            queries=queries,
            policies_applied=policies,
            preferred_sources=preferred_sources,
            disallowed_sources=disallowed_sources,
            notes=notes,
        )
    except ValidationError as exc:
        return fallback_with_reason(
            request,
            "structured strategy validation failed; using fallback strategy",
            exc,
        )
