"""Query planner for retrieval graphs."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.tool_contracts import ToolContext
from ai_core.rag.filter_spec import FilterSpec


class QueryConstraints(BaseModel):
    """Optional constraints derived from context or filters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    must_include: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    collections: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Planned retrieval query variants and constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_query: str
    queries: list[str]
    doc_type: str | None = None
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    planner: str = "rule"
    notes: str | None = None


_DOC_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "legal_contract": ("vertrag", "agreement", "contract", "nda", "mou"),
    "policy": ("policy", "guideline", "leitlinie", "richtlinie"),
    "technical": ("spec", "specification", "api", "endpoint", "schema"),
}


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_queries(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(candidate)
    return cleaned


def _infer_doc_type(query: str, doc_class: str | None) -> str | None:
    if doc_class:
        return doc_class.strip()
    lowered = query.lower()
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return doc_type
    return None


def _build_constraints(
    *,
    context: ToolContext | None,
    filters: FilterSpec | None,
) -> QueryConstraints:
    must_include: list[str] = []
    collections: list[str] = []
    document_ids: list[str] = []
    date_from = None
    date_to = None

    filters_payload: Mapping[str, Any] | None = None
    if filters is not None:
        filters_payload = filters.as_mapping()

    if filters_payload:
        raw_must_include = filters_payload.get("must_include")
        if isinstance(raw_must_include, list):
            must_include = [
                str(item).strip() for item in raw_must_include if str(item).strip()
            ]
        elif isinstance(raw_must_include, str) and raw_must_include.strip():
            must_include = [raw_must_include.strip()]

        for key in ("collection_id", "collection"):
            candidate = _coerce_str(filters_payload.get(key))
            if candidate:
                collections.append(candidate)
                break

        for key in ("document_id", "id"):
            candidate = _coerce_str(filters_payload.get(key))
            if candidate:
                document_ids.append(candidate)
                break

        date_from = _coerce_str(
            filters_payload.get("date_from") or filters_payload.get("start_date")
        )
        date_to = _coerce_str(
            filters_payload.get("date_to") or filters_payload.get("end_date")
        )

    if context is not None:
        collection_id = _coerce_str(context.business.collection_id)
        if collection_id:
            collections.append(collection_id)

    return QueryConstraints(
        must_include=must_include,
        date_from=date_from,
        date_to=date_to,
        collections=_unique_queries(collections),
        document_ids=_unique_queries(document_ids),
    )


def _build_rule_plan(
    query: str,
    *,
    context: ToolContext | None,
    doc_class: str | None,
    filters: FilterSpec | None,
) -> QueryPlan:
    doc_type = _infer_doc_type(query, doc_class)
    expansions = [query]
    if doc_type:
        expansions.append(f"{query} {doc_type.replace('_', ' ')}")
    constraints = _build_constraints(context=context, filters=filters)
    return QueryPlan(
        base_query=query,
        queries=_unique_queries(expansions),
        doc_type=doc_type,
        constraints=constraints,
        planner="rule",
    )


def _build_llm_plan(
    query: str,
    *,
    context: ToolContext | None,
    doc_class: str | None,
    filters: FilterSpec | None,
) -> QueryPlan:
    filters_payload: Mapping[str, Any] | None = None
    if filters is not None:
        filters_payload = filters.as_mapping()
    prompt = (
        "You are a query planner. Return JSON only with the schema:\n"
        "{"
        '"queries":["string"],'
        '"doc_type":"string or null",'
        '"constraints":{'
        '"must_include":["string"],'
        '"date_from":"string or null",'
        '"date_to":"string or null",'
        '"collections":["string"],'
        '"document_ids":["string"]'
        "}"
        "}\n"
        "Use 1-5 queries, include the original query. Keep expansions tight.\n"
        f"Query: {query}\n"
        f"Doc class: {doc_class or 'none'}\n"
        f"Filters: {json.dumps(filters_payload or {}, ensure_ascii=True)}\n"
    )
    metadata = {
        "tenant_id": context.scope.tenant_id if context else None,
        "case_id": context.business.case_id if context else None,
        "trace_id": context.scope.trace_id if context else None,
        "prompt_version": "rag.query_planner.v1",
    }
    response = llm_client.call(
        "analyze",
        prompt,
        metadata,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.get("text") or "{}")
    queries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(queries, list):
        raise ValueError("planner queries missing")
    doc_type = (
        _coerce_str(payload.get("doc_type")) if isinstance(payload, dict) else None
    )
    constraints_payload = (
        payload.get("constraints") if isinstance(payload, dict) else None
    )
    constraints = QueryConstraints.model_validate(constraints_payload or {})
    return QueryPlan(
        base_query=query,
        queries=_unique_queries([str(item) for item in queries]),
        doc_type=doc_type,
        constraints=constraints,
        planner="llm",
    )


def plan_query(
    query: str,
    *,
    context: ToolContext | None = None,
    doc_class: str | None = None,
    filters: FilterSpec | None = None,
) -> QueryPlan:
    base_query = (query or "").strip()
    if not base_query:
        return QueryPlan(base_query="", queries=[], planner="rule")

    mode = os.getenv("RAG_QUERY_PLANNER_MODE", "rule").strip().lower()
    if mode == "llm":
        try:
            return _build_llm_plan(
                base_query,
                context=context,
                doc_class=doc_class,
                filters=filters,
            )
        except (LlmClientError, RateLimitError, ValueError, json.JSONDecodeError):
            return _build_rule_plan(
                base_query,
                context=context,
                doc_class=doc_class,
                filters=filters,
            )

    return _build_rule_plan(
        base_query,
        context=context,
        doc_class=doc_class,
        filters=filters,
    )


__all__ = ["QueryPlan", "QueryConstraints", "plan_query"]
