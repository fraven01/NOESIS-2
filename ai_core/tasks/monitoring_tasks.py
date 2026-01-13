from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from celery import shared_task
from django.conf import settings
from django.utils import timezone
from redis import Redis

from ai_core.infra import object_store
from ai_core.infra.observability import emit_event
from ai_core.rag.vector_store import get_default_router
from common.celery import ScopedTask
from common.logging import get_logger

from .helpers.task_utils import _coerce_positive_int, _task_result

logger = get_logger(__name__)


def _is_redis_broker(url: str) -> bool:
    return url.startswith("redis://") or url.startswith("rediss://")


def _resolve_dlq_queue_key(queue_name: str) -> str:
    prefix = getattr(settings, "CELERY_REDIS_QUEUE_PREFIX", "") or ""
    return f"{prefix}{queue_name}"


def _decode_dlq_message(raw: bytes) -> Dict[str, Any] | None:
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_dead_lettered_at(payload: Mapping[str, Any]) -> Optional[float]:
    args: Any = None
    body = payload.get("body")
    if isinstance(body, (list, tuple)) and body:
        args = body[0]
    elif isinstance(body, Mapping):
        args = body.get("args")
    if args is None:
        args = payload.get("args")

    candidate: Any = None
    if isinstance(args, (list, tuple)) and args:
        candidate = args[0]
    elif isinstance(args, Mapping):
        candidate = args

    if not isinstance(candidate, Mapping):
        return None
    timestamp = candidate.get("dead_lettered_at")
    if timestamp is None:
        return None
    try:
        return float(timestamp)
    except (TypeError, ValueError):
        return None


def _is_dlq_message_expired(raw: bytes, cutoff_ts: float) -> bool:
    payload = _decode_dlq_message(raw)
    if payload is None:
        return False
    dead_lettered_at = _extract_dead_lettered_at(payload)
    if dead_lettered_at is None:
        return False
    return dead_lettered_at < cutoff_ts


@shared_task(
    base=ScopedTask,
    queue="default",
    name="ai_core.tasks.cleanup_dead_letter",
)
def cleanup_dead_letter_queue(
    *,
    max_messages: int = 1000,
    ttl_ms: Optional[int] = None,
    queue_name: str = "dead_letter",
) -> Dict[str, Any]:
    """Purge expired dead-letter tasks from a Redis-backed queue."""

    broker_url = str(getattr(settings, "CELERY_BROKER_URL", "") or "")
    if not _is_redis_broker(broker_url):
        return _task_result(
            status="success",
            data={"status": "skipped", "reason": "non_redis_broker"},
            task_name=getattr(cleanup_dead_letter_queue, "name", "cleanup_dead_letter"),
        )

    effective_ttl = ttl_ms
    if effective_ttl is None:
        effective_ttl = getattr(settings, "CELERY_DLQ_TTL_MS", 0)
    try:
        ttl_value = int(effective_ttl) if effective_ttl is not None else 0
    except (TypeError, ValueError):
        ttl_value = 0
    if ttl_value <= 0:
        return _task_result(
            status="success",
            data={"status": "skipped", "reason": "ttl_disabled"},
            task_name=getattr(cleanup_dead_letter_queue, "name", "cleanup_dead_letter"),
        )

    cutoff_ts = time.time() - (ttl_value / 1000.0)
    queue_key = _resolve_dlq_queue_key(queue_name)

    removed = 0
    kept = 0
    scanned = 0

    client = Redis.from_url(broker_url)
    scan_limit = max(0, int(max_messages))
    try:
        queue_length = int(client.llen(queue_key))
    except Exception:
        queue_length = 0
    if queue_length:
        scan_limit = min(scan_limit, queue_length)
    for _ in range(scan_limit):
        raw = client.lpop(queue_key)
        if raw is None:
            break
        scanned += 1
        if _is_dlq_message_expired(raw, cutoff_ts):
            removed += 1
            continue
        client.rpush(queue_key, raw)
        kept += 1

    logger.info(
        "dlq.cleanup.completed",
        extra={
            "queue": queue_name,
            "scanned": scanned,
            "removed": removed,
            "kept": kept,
            "ttl_ms": ttl_value,
        },
    )
    return _task_result(
        status="success",
        data={
            "status": "ok",
            "scanned": scanned,
            "removed": removed,
            "kept": kept,
        },
        task_name=getattr(cleanup_dead_letter_queue, "name", "cleanup_dead_letter"),
    )


@shared_task(
    base=ScopedTask,
    queue="default",
    name="ai_core.tasks.alert_dead_letter",
)
def alert_dead_letter_queue(
    *,
    threshold: Optional[int] = None,
    queue_name: str = "dead_letter",
) -> Dict[str, Any]:
    """Emit a structured alert when the Redis DLQ exceeds the threshold."""

    broker_url = str(getattr(settings, "CELERY_BROKER_URL", "") or "")
    if not _is_redis_broker(broker_url):
        return _task_result(
            status="success",
            data={"status": "skipped", "reason": "non_redis_broker"},
            task_name=getattr(alert_dead_letter_queue, "name", "alert_dead_letter"),
        )

    threshold_value = threshold
    if threshold_value is None:
        threshold_value = getattr(settings, "CELERY_DLQ_ALERT_THRESHOLD", 10)
    threshold_value = _coerce_positive_int(threshold_value, 10)
    if threshold_value <= 0:
        return _task_result(
            status="success",
            data={"status": "skipped", "reason": "threshold_disabled"},
            task_name=getattr(alert_dead_letter_queue, "name", "alert_dead_letter"),
        )

    queue_key = _resolve_dlq_queue_key(queue_name)
    client = Redis.from_url(broker_url)
    try:
        queue_length = int(client.llen(queue_key))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "dlq.alert.redis_error",
            extra={"queue": queue_name, "error": str(exc)},
        )
        return _task_result(
            status="error",
            data={"status": "error", "reason": "redis_error"},
            task_name=getattr(alert_dead_letter_queue, "name", "alert_dead_letter"),
            error=str(exc),
        )

    payload = {
        "queue": queue_name,
        "queue_length": queue_length,
        "threshold": threshold_value,
    }
    if queue_length > threshold_value:
        emit_event("dlq.threshold_exceeded", payload)
        logger.warning("dlq.threshold_exceeded", extra=payload)

    return _task_result(
        status="success",
        data={
            "status": "ok",
            "queue_length": queue_length,
            "threshold": threshold_value,
            "alerted": queue_length > threshold_value,
        },
        task_name=getattr(alert_dead_letter_queue, "name", "alert_dead_letter"),
    )


def _load_drift_ground_truth(path: str) -> tuple[list[dict[str, object]], str] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    raw_bytes = candidate.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    dataset_id = hashlib.sha256(raw_bytes).hexdigest()[:12]
    return [item for item in payload if isinstance(item, Mapping)], dataset_id


def _drift_metrics_path(tenant_id: str, dataset_id: str) -> str:
    tenant_segment = object_store.sanitize_identifier(tenant_id)
    dataset_segment = object_store.safe_filename(f"{dataset_id}.json")
    return "/".join(("quality", "drift", tenant_segment, dataset_segment))


def _load_previous_drift_metrics(
    tenant_id: str, dataset_id: str
) -> dict[str, object] | None:
    path = _drift_metrics_path(tenant_id, dataset_id)
    try:
        previous = object_store.read_json(path)
    except FileNotFoundError:
        return None
    if not isinstance(previous, dict):
        return None
    if previous.get("dataset_id") != dataset_id:
        return None
    return previous


def _extract_expected_ids(record: Mapping[str, object]) -> tuple[set[str], str]:
    raw_chunks = record.get("expected_chunk_ids") or record.get("expected_chunks")
    raw_docs = record.get("expected_document_ids") or record.get("expected_documents")
    chunk_ids = {str(value) for value in (raw_chunks or []) if value}
    doc_ids = {str(value) for value in (raw_docs or []) if value}
    if chunk_ids:
        return chunk_ids, "chunk_id"
    return doc_ids, "document_id"


def _evaluate_drift_queries(
    records: list[dict[str, object]],
    *,
    default_top_k: int,
) -> tuple[dict[str, dict[str, float]], int, int]:
    router = get_default_router()
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    skipped = 0

    for record in records:
        query = str(record.get("query") or "").strip()
        tenant_id = str(record.get("tenant_id") or "").strip()
        if not query or not tenant_id:
            skipped += 1
            continue

        expected_ids, id_field = _extract_expected_ids(record)
        if not expected_ids:
            skipped += 1
            continue

        try:
            top_k = int(record.get("top_k") or default_top_k)
        except (TypeError, ValueError):
            top_k = default_top_k
        if top_k <= 0:
            top_k = default_top_k

        collection_id = record.get("collection_id")
        workflow_id = record.get("workflow_id")
        case_id = record.get("case_id")

        try:
            results = router.search(
                query,
                tenant_id=tenant_id,
                case_id=str(case_id) if case_id else None,
                top_k=top_k,
                collection_id=str(collection_id) if collection_id else None,
                workflow_id=str(workflow_id) if workflow_id else None,
            )
        except Exception:
            skipped += 1
            continue

        candidate_ids: set[str] = set()
        for chunk in results:
            meta = chunk.meta or {}
            candidate_value = meta.get(id_field)
            if candidate_value:
                candidate_ids.add(str(candidate_value))

        totals[tenant_id] = totals.get(tenant_id, 0) + 1
        if expected_ids.intersection(candidate_ids):
            hits[tenant_id] = hits.get(tenant_id, 0) + 1

    metrics: dict[str, dict[str, float]] = {}
    for tenant_id, total in totals.items():
        hit_count = hits.get(tenant_id, 0)
        recall = hit_count / float(total) if total else 0.0
        metrics[tenant_id] = {
            "recall": recall,
            "hits": float(hit_count),
            "total": float(total),
        }

    return metrics, skipped, len(records)


@shared_task(
    base=ScopedTask,
    queue="default",
    name="ai_core.tasks.embedding_drift_check",
)
def embedding_drift_check(
    *,
    ground_truth_path: str | None = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Evaluate retrieval drift using ground-truth queries."""

    path = ground_truth_path
    if not path:
        path = os.getenv("RAG_DRIFT_GROUND_TRUTH_PATH")
    if not path:
        path = getattr(settings, "RAG_DRIFT_GROUND_TRUTH_PATH", None)
    if not path:
        return _task_result(
            status="success",
            data={"status": "skipped", "reason": "ground_truth_path_missing"},
            task_name=getattr(embedding_drift_check, "name", "embedding_drift_check"),
        )

    loaded = _load_drift_ground_truth(path)
    if not loaded:
        return _task_result(
            status="success",
            data={"status": "skipped", "reason": "ground_truth_unavailable"},
            task_name=getattr(embedding_drift_check, "name", "embedding_drift_check"),
        )

    records, dataset_id = loaded
    metrics, skipped, total = _evaluate_drift_queries(records, default_top_k=top_k)

    summary: dict[str, object] = {
        "status": "ok",
        "dataset_id": dataset_id,
        "total_queries": total,
        "skipped_queries": skipped,
        "tenants": list(metrics.keys()),
    }

    for tenant_id, tenant_metrics in metrics.items():
        recall = float(tenant_metrics.get("recall", 0.0))
        previous = _load_previous_drift_metrics(tenant_id, dataset_id)
        previous_recall = None
        if previous:
            try:
                previous_recall = float(previous.get("recall", 0.0))
            except (TypeError, ValueError):
                previous_recall = None
        delta = recall - previous_recall if previous_recall is not None else None

        payload = {
            "tenant_id": tenant_id,
            "dataset_id": dataset_id,
            "recall": recall,
            "previous_recall": previous_recall,
            "delta": delta,
            "total_queries": tenant_metrics.get("total"),
            "hits": tenant_metrics.get("hits"),
        }
        logger.info("rag.drift.recall", extra=payload)

        if delta is not None and delta <= -0.1:
            emit_event("rag.drift.recall_drop", payload)
            logger.warning("rag.drift.recall_drop", extra=payload)

        metrics_path = _drift_metrics_path(tenant_id, dataset_id)
        object_store.write_json(
            metrics_path,
            {
                "dataset_id": dataset_id,
                "recall": recall,
                "total_queries": tenant_metrics.get("total"),
                "hits": tenant_metrics.get("hits"),
                "computed_at": timezone.now().isoformat(),
            },
        )

    return _task_result(
        status="success",
        data=summary,
        task_name=getattr(embedding_drift_check, "name", "embedding_drift_check"),
    )
