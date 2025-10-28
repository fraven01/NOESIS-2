"""Ingestion payload planning based on normalized documents and delta decisions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Protocol

from documents.contracts import NormalizedDocument
from documents.normalization import document_parser_stats, resolve_provider_reference
from documents.repository import DEFAULT_LIFECYCLE_STORE

from .contracts import Decision
from .delta import DeltaDecision
from .errors import CrawlerError, ErrorClass
from .retire import LifecycleDecision, LifecycleState

from ai_core.rag.ingestion_contracts import (
    IngestionAction,
    build_crawler_ingestion_payload,
)

IngestionStatus = IngestionAction


_LIFECYCLE_STORE = DEFAULT_LIFECYCLE_STORE


class CrawlerIngestionAdapter(Protocol):
    """Adapter interface for crawler-specific ingestion metadata."""

    def build_metadata(
        self, document: NormalizedDocument
    ) -> Mapping[str, object]:  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class DefaultCrawlerIngestionAdapter:
    """Default adapter forwarding crawler metadata to ingestion services."""

    def build_metadata(self, document: NormalizedDocument) -> Mapping[str, object]:
        external = resolve_provider_reference(document)
        parser_stats = document_parser_stats(document)
        metadata: Dict[str, object] = {
            "canonical_source": external.canonical_source,
            "provider": external.provider,
            "provider_tags": dict(external.provider_tags),
            "parser_stats": dict(parser_stats),
            "origin_uri": document.meta.origin_uri,
            "title": document.meta.title,
            "language": document.meta.language,
            "media_type": getattr(document.blob, "media_type", None),
            "tags": list(document.meta.tags),
        }
        return MappingProxyType(
            {key: value for key, value in metadata.items() if value is not None}
        )


def build_ingestion_decision(
    document: NormalizedDocument,
    delta: DeltaDecision,
    *,
    case_id: str,
    retire: bool = False,
    lifecycle: Optional[LifecycleDecision] = None,
    embedding_profile: Optional[str] = None,
    adapter: Optional[CrawlerIngestionAdapter] = None,
) -> Decision:
    """Compose an ingestion decision from normalized document and delta information."""

    normalized_case_id = _require_identifier(case_id, "case_id")
    lifecycle_decision = _resolve_lifecycle(lifecycle, retire)
    adapter_impl = adapter or DefaultCrawlerIngestionAdapter()
    adapter_metadata = adapter_impl.build_metadata(document)

    action = (
        IngestionAction.RETIRE
        if lifecycle_decision.should_retire
        else IngestionAction.UPSERT
    )

    payload = build_crawler_ingestion_payload(
        document=document,
        signatures=delta.signatures,
        case_id=normalized_case_id,
        action=action,
        lifecycle_state=lifecycle_decision.state,
        policy_events=lifecycle_decision.policy_events,
        adapter_metadata=adapter_metadata,
        embedding_profile=embedding_profile,
        delta_status=delta.status,
    )

    reason = (
        lifecycle_decision.reason if lifecycle_decision.should_retire else delta.reason
    )
    attributes = dict(payload.as_mapping())
    attributes["lifecycle_state"] = lifecycle_decision.state
    _LIFECYCLE_STORE.record_document_state(
        tenant_id=document.ref.tenant_id,
        document_id=document.ref.document_id,
        workflow_id=document.ref.workflow_id,
        state=lifecycle_decision.state.value,
        reason=lifecycle_decision.reason,
        policy_events=lifecycle_decision.policy_events,
        changed_at=document.created_at,
    )
    return Decision(payload.action.value, reason, attributes)


def _require_identifier(value: Optional[str], field: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError(f"{field}_required")
    return candidate


def _resolve_lifecycle(
    lifecycle: Optional[LifecycleDecision], retire: bool
) -> LifecycleDecision:
    if lifecycle is not None:
        return lifecycle
    if retire:
        return LifecycleDecision(LifecycleState.RETIRED, "retired")
    return LifecycleDecision(LifecycleState.ACTIVE, "active")


def build_ingestion_error(
    *,
    decision: Optional[Decision],
    reason: str,
    error_code: Optional[str] = None,
    status_code: Optional[int] = None,
) -> CrawlerError:
    """Create a standardized ingestion failure payload for observability."""

    attributes: Dict[str, object] = {}
    if error_code:
        attributes["error_code"] = error_code

    adapter_metadata: Mapping[str, object] | None = None
    if decision is not None:
        adapter_metadata = decision.attributes.get("adapter_metadata")
    source = (
        adapter_metadata.get("canonical_source")
        if isinstance(adapter_metadata, Mapping)
        else None
    )
    provider = (
        adapter_metadata.get("provider")
        if isinstance(adapter_metadata, Mapping)
        else None
    )

    return CrawlerError(
        error_class=ErrorClass.INGESTION_FAILURE,
        reason=reason,
        source=source,
        provider=provider,
        status_code=status_code,
        attributes=attributes,
    )
