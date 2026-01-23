from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from celery import shared_task
from django.db import transaction
from django_tenants.utils import get_public_schema_name, schema_context

from ai_core.infra.observability import observe_span
from ai_core.models import RagFeedbackEvent
from ai_core.rag.feedback import update_weight_profiles


@shared_task(queue="default", name="ai_core.tasks.rag_feedback.collect")
@observe_span(name="rag.feedback.collect")
def record_rag_feedback_events(events: Sequence[Mapping[str, Any]]) -> int:
    if not events:
        return 0
    from customers.tenant_context import TenantContext

    records_by_schema: dict[str, list[RagFeedbackEvent]] = {}
    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            tenant_id = event.get("tenant_id")
            feedback_type = event.get("feedback_type")
            if not tenant_id or not feedback_type:
                continue
            tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
            if tenant is None or not tenant.schema_name:
                continue
            schema_name = tenant.schema_name
            records_by_schema.setdefault(schema_name, []).append(
                RagFeedbackEvent(
                    tenant_id=str(tenant_id),
                    case_id=event.get("case_id"),
                    collection_id=event.get("collection_id"),
                    workflow_id=event.get("workflow_id"),
                    thread_id=event.get("thread_id"),
                    trace_id=event.get("trace_id"),
                    quality_mode=event.get("quality_mode") or "standard",
                    feedback_type=str(feedback_type),
                    query_text=event.get("query_text"),
                    source_id=event.get("source_id"),
                    source_label=event.get("source_label"),
                    document_id=event.get("document_id"),
                    chunk_id=event.get("chunk_id"),
                    relevance_score=event.get("relevance_score"),
                    feature_payload=event.get("feature_payload"),
                    metadata=event.get("metadata"),
                )
            )
    if not records_by_schema:
        return 0
    total = 0
    for schema_name, records in records_by_schema.items():
        with schema_context(schema_name):
            with transaction.atomic():
                RagFeedbackEvent.objects.bulk_create(records, batch_size=250)
        total += len(records)
    return total


@shared_task(queue="default", name="ai_core.tasks.rag_feedback.update_weights")
@observe_span(name="rag.feedback.update_weights")
def update_rag_rerank_weights() -> dict[str, int]:
    window_days_raw = os.getenv("RAG_RERANK_FEEDBACK_WINDOW_DAYS", "7")
    try:
        window_days = int(window_days_raw)
    except (TypeError, ValueError):
        window_days = 7
    window_days = max(1, window_days)

    results = update_weight_profiles(window_days=window_days)
    updated = sum(1 for item in results if item.updated)
    return {"updated": updated, "total": len(results)}


__all__ = ["record_rag_feedback_events", "update_rag_rerank_weights"]
