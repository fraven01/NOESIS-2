"""Case domain helpers."""

from __future__ import annotations

from typing import Final

from common.logging import get_logger
from customers.models import Tenant
from documents.models import DocumentIngestionRun

from cases import models


_CASE_STATUS_OPEN: Final[str] = models.Case.Status.OPEN
_INGESTION_EVENT_SOURCE: Final[str] = "ingestion"
_INGESTION_STATUS_EVENT_TYPES: Final[dict[str, str]] = {
    "queued": "ingestion_run_queued",
    "running": "ingestion_run_started",
    "succeeded": "ingestion_run_completed",
    "failed": "ingestion_run_failed",
}
_INGESTION_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    "status",
    "queued_at",
    "started_at",
    "finished_at",
    "duration_ms",
    "inserted_documents",
    "replaced_documents",
    "skipped_documents",
    "inserted_chunks",
)

log = get_logger(__name__)


def _apply_case_phase(case: models.Case) -> None:
    from cases.lifecycle import apply_lifecycle_definition

    apply_lifecycle_definition(
        case,
        list(case.events.order_by("created_at").only("id", "event_type", "created_at")),
    )


def _normalise_case_id(case_id: str) -> str:
    candidate = (case_id or "").strip()
    if not candidate:
        raise ValueError("case_id must not be empty")
    return candidate


def get_or_create_case_for(tenant: Tenant, case_id: str) -> models.Case:
    """Return the case for *tenant* and *case_id*, creating it when missing."""

    normalized_case_id = _normalise_case_id(case_id)
    case, created = models.Case.objects.get_or_create(
        tenant=tenant,
        external_id=normalized_case_id,
        defaults={"status": _CASE_STATUS_OPEN},
    )
    if created:
        log.info(
            "case.created",
            extra={
                "tenant": str(getattr(tenant, "schema_name", tenant.pk)),
                "case_external_id": normalized_case_id,
                "case_status": case.status,
                "case_phase": case.phase or "",
                "trigger_event_type": "case.create",
            },
        )
    return case


def _build_ingestion_payload(ingestion_run: DocumentIngestionRun) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": ingestion_run.run_id,
    }
    for field in _INGESTION_PAYLOAD_FIELDS:
        value = getattr(ingestion_run, field, None)
        if value in (None, ""):
            continue
        payload[field] = value
    invalid_ids = getattr(ingestion_run, "invalid_document_ids", None)
    if invalid_ids:
        payload["invalid_document_ids"] = invalid_ids
    document_ids = getattr(ingestion_run, "document_ids", None)
    if document_ids:
        payload["document_ids"] = document_ids
    error = getattr(ingestion_run, "error", None)
    if error:
        payload["error"] = error
    return payload


def record_ingestion_case_event(
    tenant: Tenant, case_id: str, ingestion_run: DocumentIngestionRun
) -> models.CaseEvent:
    """Create a case event reflecting the latest ingestion run state."""

    case = get_or_create_case_for(tenant, case_id)
    normalized_status = (ingestion_run.status or "").strip().lower()
    event_type = _INGESTION_STATUS_EVENT_TYPES.get(
        normalized_status, "ingestion_run_updated"
    )
    payload = _build_ingestion_payload(ingestion_run)

    event = models.CaseEvent.objects.create(
        case=case,
        tenant=tenant,
        event_type=event_type,
        source=_INGESTION_EVENT_SOURCE,
        ingestion_run=ingestion_run,
        collection_id=getattr(ingestion_run, "collection_id", "") or "",
        trace_id=getattr(ingestion_run, "trace_id", "") or "",
        payload=payload,
    )
    try:
        _apply_case_phase(case)
    except Exception:
        log.exception(
            "case_lifecycle_apply_failed",
            extra={"tenant": tenant.schema_name, "case": case.external_id},
        )
    return event
