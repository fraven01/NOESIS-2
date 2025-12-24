"""Case domain helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Mapping

from common.logging import get_logger
from customers.models import Tenant

if TYPE_CHECKING:
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


class CaseNotFoundError(Exception):
    """Raised when a requested case ID does not exist."""


def resolve_case(tenant: Tenant, case_id: str | None) -> models.Case | None:
    """Resolve a case by ID, ensuring it exists. Returns None if case_id is empty."""
    if not case_id:
        return None

    normalized_case_id = _normalise_case_id(case_id)
    try:
        return models.Case.objects.get(tenant=tenant, external_id=normalized_case_id)
    except models.Case.DoesNotExist:
        raise CaseNotFoundError(
            f"Case {normalized_case_id} not found for tenant {getattr(tenant, 'schema_name', tenant.pk)}"
        )


def ensure_case(
    tenant: Tenant,
    case_id: str,
    *,
    title: str | None = None,
    metadata: Mapping[str, object] | None = None,
    reopen_closed: bool = False,
) -> models.Case:
    """Ensure a case exists for the tenant, creating it if missing."""
    normalized_case_id = _normalise_case_id(case_id)
    defaults: dict[str, object] = {"status": _CASE_STATUS_OPEN}
    if title is not None:
        defaults["title"] = title
    if metadata is not None:
        defaults["metadata"] = dict(metadata)

    case, created = models.Case.objects.get_or_create(
        tenant=tenant,
        external_id=normalized_case_id,
        defaults=defaults,
    )

    if not created and reopen_closed and case.status != _CASE_STATUS_OPEN:
        case.status = _CASE_STATUS_OPEN
        case.closed_at = None
        case.save(update_fields=["status", "closed_at", "updated_at"])
        log.info(
            "case.reopened",
            extra={
                "tenant": getattr(tenant, "schema_name", tenant.pk),
                "case": case.external_id,
            },
        )

    return case


def _build_ingestion_payload(
    ingestion_run: "DocumentIngestionRun",
) -> dict[str, object]:
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
    tenant: Tenant, case_id: str | None, ingestion_run: "DocumentIngestionRun"
) -> models.CaseEvent | None:
    """Create a case event reflecting the latest ingestion run state."""

    case = resolve_case(tenant, case_id)
    if not case:
        return None

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
