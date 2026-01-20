from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from celery import shared_task
from django.db import transaction

from ai_core.infra.observability import observe_span
from ai_core.models import RagFeedbackEvent
from ai_core.rag.feedback import update_weight_profiles


@shared_task(queue="default", name="ai_core.tasks.rag_feedback.collect")
@observe_span(name="rag.feedback.collect")
def record_rag_feedback_events(events: Sequence[Mapping[str, Any]]) -> int:
    if not events:
        return 0
    records: list[RagFeedbackEvent] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        tenant_id = event.get("tenant_id")
        feedback_type = event.get("feedback_type")
        if not tenant_id or not feedback_type:
            continue
        records.append(
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
    if not records:
        return 0
    with transaction.atomic():
        RagFeedbackEvent.objects.bulk_create(records, batch_size=250)
    return len(records)


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
