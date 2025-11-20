"""Case lifecycle helpers derived from graph transitions and definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from django.db import transaction
from django.utils import timezone

from common.logging import get_logger
from customers.models import Tenant

from pydantic import ValidationError as PydanticValidationError

from cases import models
from cases.contracts import (
    CaseLifecycleDefinition,
    DEFAULT_CASE_LIFECYCLE_DEFINITION,
    parse_case_lifecycle_definition,
)
from cases.services import get_or_create_case_for

log = get_logger(__name__)

_COLLECTION_SEARCH_SOURCE = "graph:CollectionSearchGraph"
_COLLECTION_SEARCH_GRAPH_NAME = "collection_search"
_EVENT_PREFIX = "collection_search:"
_MAX_EVENTS_FOR_APPLY = 250


@dataclass(frozen=True, slots=True)
class CaseLifecycleUpdateResult:
    case: models.Case
    tenant: Tenant
    event_types: tuple[str, ...]
    trace_id: str
    workflow_id: str
    collection_scope: str


def _normalise_mapping(candidate: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return candidate if isinstance(candidate, Mapping) else {}


def _tenant_for_id(tenant_id: str | None) -> Tenant | None:
    from customers.tenant_context import TenantContext

    return TenantContext.resolve_identifier(tenant_id)


def _coerce_transitions(transitions: object) -> list[Mapping[str, Any]]:
    if isinstance(transitions, Mapping):
        values = transitions.values()
    elif isinstance(transitions, Sequence):
        values = transitions
    else:
        values = []
    cleaned: list[Mapping[str, Any]] = []
    for entry in values:
        if isinstance(entry, Mapping):
            cleaned.append(entry)
    return cleaned


def _transition_name(entry: Mapping[str, Any]) -> str:
    name = str(entry.get("node") or "").strip()
    if name:
        return name
    name = str(entry.get("transition") or "").strip()
    return name


def _definition_for_tenant(tenant: Tenant) -> CaseLifecycleDefinition:
    definition = getattr(tenant, "case_lifecycle_definition", None)
    if not definition:
        return DEFAULT_CASE_LIFECYCLE_DEFINITION.model_copy(deep=True)
    try:
        parsed = parse_case_lifecycle_definition(definition)
    except PydanticValidationError:
        log.warning(
            "case_lifecycle_invalid_definition",
            extra={"tenant": str(getattr(tenant, "schema_name", tenant.pk))},
        )
        return DEFAULT_CASE_LIFECYCLE_DEFINITION.model_copy(deep=True)
    if parsed is None:
        return DEFAULT_CASE_LIFECYCLE_DEFINITION.model_copy(deep=True)
    return parsed


def _resolve_transitions(
    definition: CaseLifecycleDefinition,
) -> dict[str, list[tuple[str | None, str]]]:
    mapping: dict[str, list[tuple[str | None, str]]] = {}
    for transition in definition.transitions:
        to_phase = (transition.to_phase or "").strip()
        if not to_phase:
            continue
        for event in transition.trigger_events:
            key = (event or "").strip()
            if not key:
                continue
            mapping.setdefault(key, []).append((transition.from_phase, to_phase))
    return mapping


def _ordered_events(events: Sequence[models.CaseEvent]) -> list[models.CaseEvent]:
    def _sort_key(event: models.CaseEvent) -> tuple[object, int]:
        created = event.created_at or timezone.now()
        pk = getattr(event, "pk", 0) or 0
        return (created, int(pk))

    return sorted(events, key=_sort_key)


def apply_lifecycle_definition(
    case: models.Case, events: Sequence[models.CaseEvent]
) -> None:
    """Derive and persist the latest phase for *case* based on *events*."""

    if not events:
        return

    definition = _definition_for_tenant(case.tenant)
    transition_index = _resolve_transitions(definition)
    if not transition_index:
        return

    ordered_events = _ordered_events(events)
    current_phase = (case.phase or "").strip() or None
    new_phase = current_phase
    updated_by_event: str | None = None

    for event in ordered_events[-_MAX_EVENTS_FOR_APPLY:]:
        event_type = (event.event_type or "").strip()
        if not event_type:
            continue
        transitions = transition_index.get(event_type)
        if not transitions:
            continue
        for from_phase, to_phase in transitions:
            if from_phase is not None and from_phase != new_phase:
                continue
            if not to_phase:
                continue
            if new_phase != to_phase:
                new_phase = to_phase
                updated_by_event = event.event_type or updated_by_event
            break

    final_phase = new_phase or ""
    if final_phase != (case.phase or ""):
        case.phase = final_phase
        case.save(update_fields=["phase", "updated_at"])
        log.info(
            "case.phase.updated",
            extra={
                "tenant": str(getattr(case.tenant, "schema_name", case.tenant_id)),
                "case_external_id": case.external_id,
                "case_phase": final_phase,
                "trigger_event_type": updated_by_event or "",
            },
        )


def update_case_from_collection_search(
    tenant_id: str | None, case_id: str, graph_state: Mapping[str, Any] | None
) -> CaseLifecycleUpdateResult | None:
    """Persist case lifecycle events derived from the CollectionSearch graph."""

    tenant = _tenant_for_id(tenant_id)
    if tenant is None:
        log.debug(
            "case_lifecycle_missing_tenant",
            extra={"tenant_id": tenant_id, "case": case_id},
        )
        return None

    try:
        case = get_or_create_case_for(tenant, case_id)
    except ValueError:
        log.debug(
            "case_lifecycle_invalid_case",
            extra={"tenant_id": tenant_id, "case": case_id},
        )
        return None

    safe_state = _normalise_mapping(graph_state)
    transitions = _coerce_transitions(safe_state.get("transitions"))
    context = _normalise_mapping(safe_state.get("context"))
    workflow_id = str(context.get("workflow_id") or "")
    trace_id = str(context.get("trace_id") or "")
    collection_scope = str(context.get("collection_scope") or "")
    if not collection_scope:
        input_section = _normalise_mapping(safe_state.get("input"))
        collection_scope = str(input_section.get("collection_scope") or "")

    if not transitions:
        return None
    created_events = False
    recorded_types: list[str] = []
    with transaction.atomic():
        for entry in transitions:
            name = _transition_name(entry)
            if not name:
                continue
            payload = dict(entry)
            event_type = f"{_EVENT_PREFIX}{name}"
            models.CaseEvent.objects.create(
                case=case,
                tenant=tenant,
                event_type=event_type,
                source=_COLLECTION_SEARCH_SOURCE,
                graph_name=_COLLECTION_SEARCH_GRAPH_NAME,
                workflow_id=workflow_id,
                trace_id=trace_id,
                collection_id=collection_scope,
                payload=payload,
            )
            created_events = True
            recorded_types.append(event_type)

    if created_events:
        try:
            apply_lifecycle_definition(
                case,
                list(
                    case.events.order_by("created_at").only(
                        "id", "event_type", "created_at"
                    )
                ),
            )
        except Exception:
            log.exception(
                "case_lifecycle_apply_failed",
                extra={"tenant": str(tenant.pk), "case": case.external_id},
            )
        return CaseLifecycleUpdateResult(
            case=case,
            tenant=tenant,
            event_types=tuple(recorded_types),
            trace_id=trace_id,
            workflow_id=workflow_id,
            collection_scope=collection_scope,
        )

    return None


__all__ = [
    "CaseLifecycleUpdateResult",
    "apply_lifecycle_definition",
    "update_case_from_collection_search",
]
