from __future__ import annotations

import os
import time
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from django_tenants.utils import get_public_schema_name, schema_context
from django.utils import timezone

from ai_core.models import RagFeedbackEvent, RagRerankWeight
from ai_core.tool_contracts import ToolContext
from ai_core.rag.rerank_features import (
    extract_rerank_features,
    get_static_weight_profile,
)

_WEIGHT_KEYS = (
    "parent_relevance",
    "section_match",
    "confidence",
    "adjacency_bonus",
    "doc_type_match",
)

_LEARNED_CACHE_TTL_SECONDS = 300
_LEARNED_CACHE: dict[tuple[str, str], tuple[float, dict[str, float]]] = {}


def _schema_context_for_tenant_id(tenant_id: object):
    from customers.tenant_context import TenantContext

    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
    if tenant is None or not tenant.schema_name:
        return nullcontext()
    return schema_context(tenant.schema_name)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def should_collect_feedback() -> bool:
    return _env_flag("RAG_FEEDBACK_ENABLED", default=False)


def _resolve_quality_mode(context: ToolContext | None) -> str:
    if context is None:
        return "standard"
    meta = getattr(context, "metadata", {})
    if isinstance(meta, Mapping):
        candidate = meta.get("quality_mode")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return "standard"


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


def _resolve_source_label(source: object) -> str | None:
    if isinstance(source, Mapping):
        raw = source.get("label") or source.get("id")
    else:
        raw = getattr(source, "label", None) or getattr(source, "id", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _resolve_source_score(source: object) -> float | None:
    raw = None
    if isinstance(source, Mapping):
        raw = source.get("relevance_score")
    else:
        raw = getattr(source, "relevance_score", None)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _build_feature_map(
    snippets: Sequence[Mapping[str, Any]],
    *,
    context: ToolContext,
    quality_mode: str,
) -> dict[str, dict[str, float]]:
    features = extract_rerank_features(
        snippets,
        context=context,
        quality_mode=quality_mode,
    )
    return {feature.chunk_id: feature.to_dict() for feature in features}


def _build_snippet_lookup(
    snippets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Mapping[str, Any]]]:
    label_to_chunk: dict[str, str] = {}
    chunk_to_snippet: dict[str, Mapping[str, Any]] = {}
    for index, snippet in enumerate(snippets):
        if not isinstance(snippet, Mapping):
            continue
        meta = snippet.get("meta")
        if isinstance(meta, Mapping):
            chunk_id = meta.get("chunk_id") or meta.get("document_id")
        else:
            chunk_id = snippet.get("id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            chunk_id = f"snippet-{index}"
        chunk_id = chunk_id.strip()
        label = _resolve_snippet_label(snippet, index)
        label_to_chunk[label] = chunk_id
        chunk_to_snippet[chunk_id] = snippet
    return label_to_chunk, chunk_to_snippet


def enqueue_used_source_feedback(
    *,
    context: ToolContext,
    query_text: str | None,
    snippets: Sequence[Mapping[str, Any]],
    used_sources: Sequence[object],
) -> None:
    if not should_collect_feedback():
        return
    if not used_sources or not snippets:
        return

    quality_mode = _resolve_quality_mode(context)
    label_to_chunk, chunk_to_snippet = _build_snippet_lookup(snippets)
    feature_map = _build_feature_map(
        snippets,
        context=context,
        quality_mode=quality_mode,
    )

    events: list[dict[str, Any]] = []
    for source in used_sources:
        label = _resolve_source_label(source)
        if not label:
            continue
        chunk_id = label_to_chunk.get(label) or label
        snippet = chunk_to_snippet.get(chunk_id)
        meta = snippet.get("meta") if isinstance(snippet, Mapping) else None
        document_id = None
        source_id = None
        if isinstance(meta, Mapping):
            document_id = meta.get("document_id")
            source_id = meta.get("chunk_id") or meta.get("document_id")
        if not isinstance(source_id, str):
            source_id = label
        events.append(
            {
                "tenant_id": context.scope.tenant_id,
                "case_id": context.business.case_id,
                "collection_id": context.business.collection_id,
                "workflow_id": context.business.workflow_id,
                "thread_id": context.business.thread_id,
                "trace_id": context.scope.trace_id,
                "quality_mode": quality_mode,
                "feedback_type": RagFeedbackEvent.FEEDBACK_USED_SOURCE,
                "query_text": query_text,
                "source_id": source_id,
                "source_label": label,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "relevance_score": _resolve_source_score(source),
                "feature_payload": feature_map.get(chunk_id),
            }
        )

    if not events:
        return
    from ai_core.tasks.rag_feedback_tasks import record_rag_feedback_events

    record_rag_feedback_events.delay(events)


def enqueue_click_feedback(
    *,
    context: ToolContext,
    payload: Mapping[str, Any],
) -> None:
    if not should_collect_feedback():
        return
    quality_mode = _resolve_quality_mode(context)
    event = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "collection_id": context.business.collection_id,
        "workflow_id": context.business.workflow_id,
        "thread_id": context.business.thread_id,
        "trace_id": context.scope.trace_id,
        "quality_mode": quality_mode,
        "feedback_type": RagFeedbackEvent.FEEDBACK_CLICK,
        "query_text": payload.get("query_text"),
        "source_id": payload.get("source_id"),
        "source_label": payload.get("source_label"),
        "document_id": payload.get("document_id"),
        "chunk_id": payload.get("chunk_id"),
        "relevance_score": payload.get("relevance_score"),
        "metadata": {
            "url": payload.get("url"),
            "ui": payload.get("ui"),
        },
    }
    from ai_core.tasks.rag_feedback_tasks import record_rag_feedback_events

    record_rag_feedback_events.delay([event])


def _normalize_weights(weights: Mapping[str, Any]) -> dict[str, float] | None:
    filtered: dict[str, float] = {}
    for key in _WEIGHT_KEYS:
        value = weights.get(key)
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if candidate < 0:
            continue
        filtered[key] = candidate
    if not filtered:
        return None
    total = sum(filtered.values())
    if total <= 0:
        return None
    return {key: filtered.get(key, 0.0) / total for key in _WEIGHT_KEYS}


def get_learned_weight_profile(
    tenant_id: str, quality_mode: str
) -> dict[str, float] | None:
    cache_key = (tenant_id, quality_mode)
    cached = _LEARNED_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) <= _LEARNED_CACHE_TTL_SECONDS:
        return dict(cached[1])

    with _schema_context_for_tenant_id(tenant_id):
        record = (
            RagRerankWeight.objects.filter(
                tenant_id=tenant_id,
                quality_mode=quality_mode,
            )
            .only("weights")
            .first()
        )
    if not record:
        return None
    normalized = _normalize_weights(record.weights or {})
    if normalized is None:
        return None
    _LEARNED_CACHE[cache_key] = (time.time(), normalized)
    return dict(normalized)


def learn_weight_profile(
    events: Sequence[RagFeedbackEvent],
    *,
    base_weights: Mapping[str, float],
) -> tuple[dict[str, float] | None, int]:
    sums = {key: 0.0 for key in _WEIGHT_KEYS}
    count = 0
    for event in events:
        payload = event.feature_payload
        if not isinstance(payload, Mapping):
            continue
        weight = 1.0
        if event.feedback_type == RagFeedbackEvent.FEEDBACK_CLICK:
            weight = float(os.getenv("RAG_RERANK_CLICK_WEIGHT", "2.0"))
        for key in _WEIGHT_KEYS:
            try:
                sums[key] += float(payload.get(key, 0.0)) * weight
            except (TypeError, ValueError):
                continue
        count += 1

    if count <= 0:
        return None, 0
    total = sum(sums.values())
    if total <= 0:
        return None, count
    normalized = {key: sums[key] / total for key in _WEIGHT_KEYS}
    alpha_raw = os.getenv("RAG_RERANK_WEIGHT_ALPHA", "0.5")
    try:
        alpha = float(alpha_raw)
    except (TypeError, ValueError):
        alpha = 0.5
    alpha = max(0.0, min(1.0, alpha))
    blended = {
        key: alpha * normalized.get(key, 0.0)
        + (1.0 - alpha) * base_weights.get(key, 0.0)
        for key in _WEIGHT_KEYS
    }
    blended_total = sum(blended.values())
    if blended_total <= 0:
        return None, count
    return {key: blended[key] / blended_total for key in _WEIGHT_KEYS}, count


@dataclass(frozen=True)
class WeightUpdateResult:
    tenant_id: str
    quality_mode: str
    sample_count: int
    updated: bool


def update_weight_profiles(*, window_days: int) -> list[WeightUpdateResult]:
    public_schema = get_public_schema_name()
    from customers.models import Tenant

    with schema_context(public_schema):
        tenants = list(Tenant.objects.exclude(schema_name=public_schema))

    results: list[WeightUpdateResult] = []
    for tenant in tenants:
        with schema_context(tenant.schema_name):
            results.extend(_update_weight_profiles_in_schema(window_days=window_days))
    return results


def _update_weight_profiles_in_schema(*, window_days: int) -> list[WeightUpdateResult]:
    since = timezone.now() - timedelta(days=window_days)
    base_qs = RagFeedbackEvent.objects.filter(
        created_at__gte=since,
        feature_payload__isnull=False,
    )
    results: list[WeightUpdateResult] = []
    groups = base_qs.values_list("tenant_id", "quality_mode").distinct()
    for tenant_id, quality_mode in groups:
        events = list(base_qs.filter(tenant_id=tenant_id, quality_mode=quality_mode))
        base_weights = get_static_weight_profile(quality_mode)
        learned, sample_count = learn_weight_profile(events, base_weights=base_weights)
        if learned is None:
            results.append(
                WeightUpdateResult(
                    tenant_id=tenant_id,
                    quality_mode=quality_mode,
                    sample_count=sample_count,
                    updated=False,
                )
            )
            continue
        RagRerankWeight.objects.update_or_create(
            tenant_id=tenant_id,
            quality_mode=quality_mode,
            defaults={
                "weights": learned,
                "sample_count": sample_count,
                "source": "learned",
            },
        )
        _LEARNED_CACHE.pop((tenant_id, quality_mode), None)
        results.append(
            WeightUpdateResult(
                tenant_id=tenant_id,
                quality_mode=quality_mode,
                sample_count=sample_count,
                updated=True,
            )
        )
    return results


__all__ = [
    "enqueue_used_source_feedback",
    "enqueue_click_feedback",
    "get_learned_weight_profile",
    "update_weight_profiles",
]
