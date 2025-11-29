"""Integration helpers between ingestion runs and case events."""

from __future__ import annotations

from typing import Optional

from django.db import DatabaseError

from common.logging import get_logger

from cases.models import CaseEvent
from cases.services import record_ingestion_case_event
from customers.models import Tenant
from documents.models import DocumentIngestionRun

from .infra.observability import emit_event


log = get_logger(__name__)


def _resolve_tenant(tenant_identifier: str) -> Optional[Tenant]:
    from customers.tenant_context import TenantContext

    return TenantContext.resolve_identifier(tenant_identifier, allow_pk=True)


def _load_ingestion_run(tenant_id: str, case_id: str | None) -> Optional[DocumentIngestionRun]:
    try:
        return DocumentIngestionRun.objects.get(tenant_id=tenant_id, case=case_id)
    except DocumentIngestionRun.DoesNotExist:
        return None
    except DatabaseError:
        raise


def emit_ingestion_case_event(
    tenant_identifier: str,
    case_id: str | None,
    *,
    run_id: str | None = None,
    context: str | None = None,
) -> None:
    """Record a case event for the current ingestion run if possible."""

    tenant = _resolve_tenant(tenant_identifier)
    if tenant is None:
        log.debug(
            "case_event_tenant_missing",
            extra={"tenant": tenant_identifier, "case": case_id, "context": context},
        )
        return

    try:
        ingestion_run = _load_ingestion_run(tenant_identifier, case_id)
    except DatabaseError:
        log.exception(
            "case_event_ingestion_lookup_failed",
            extra={"tenant": tenant_identifier, "case": case_id, "context": context},
        )
        return

    if ingestion_run is None:
        log.debug(
            "case_event_ingestion_missing",
            extra={"tenant": tenant_identifier, "case": case_id, "context": context},
        )
        return

    if run_id and ingestion_run.run_id != run_id:
        log.debug(
            "case_event_run_mismatch",
            extra={
                "tenant": tenant_identifier,
                "case": case_id,
                "context": context,
                "expected_run_id": run_id,
                "actual_run_id": ingestion_run.run_id,
            },
        )
        return

    try:
        case_event = record_ingestion_case_event(tenant, case_id, ingestion_run)
    except Exception:  # pragma: no cover - defensive logging
        log.exception(
            "case_event_record_failed",
            extra={
                "tenant": tenant_identifier,
                "case": case_id,
                "context": context,
                "run_id": run_id or ingestion_run.run_id,
            },
        )
        return

    _emit_case_observability_event(case_event)


def _emit_case_observability_event(case_event: CaseEvent) -> None:
    case = case_event.case
    ingestion_run = case_event.ingestion_run
    payload = {
        "tenant_id": str(case.tenant_id),
        "case_id": case.external_id,
        "case_status": case.status,
        "case_phase": case.phase or "",
        "case_event_type": case_event.event_type,
        "source": case_event.source,
        "collection_scope": case_event.collection_id or "",
    }
    if case_event.trace_id:
        payload["trace_id"] = case_event.trace_id
    if ingestion_run is not None:
        payload.setdefault("trace_id", ingestion_run.trace_id or "")
        payload["ingestion_run_id"] = ingestion_run.run_id
        if ingestion_run.collection_id:
            payload.setdefault("collection_scope", ingestion_run.collection_id)
    if case_event.payload:
        payload["event_payload"] = case_event.payload
    emit_event("case.lifecycle.ingestion", payload)


__all__ = ["emit_ingestion_case_event"]
