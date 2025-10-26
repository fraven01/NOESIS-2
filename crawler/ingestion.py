"""Ingestion payload planning based on normalized documents and delta decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from .delta import DeltaDecision, DeltaStatus
from .errors import CrawlerError, ErrorClass
from .normalizer import NormalizedDocument
from .retire import LifecycleDecision, LifecycleState


class IngestionStatus(str, Enum):
    """Supported ingestion actions for downstream processing."""

    UPSERT = "upsert"
    SKIP = "skip"
    RETIRE = "retire"


@dataclass(frozen=True)
class IngestionPayload:
    """Structured payload forwarded to the ingestion entrypoint."""

    tenant_id: str
    case_id: str
    workflow_id: str
    document_id: str
    content_hash: str
    external_id: str
    provider: str
    canonical_source: str
    origin_uri: str
    source: str
    media_type: Optional[str]
    title: Optional[str]
    language: Optional[str]
    tags: Tuple[str, ...] = ()
    parser_stats: Mapping[str, object] = field(default_factory=dict)
    provider_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tenant = _require_identifier(self.tenant_id, "tenant_id")
        case = _require_identifier(self.case_id, "case_id")
        workflow = _require_identifier(self.workflow_id, "workflow_id")
        document = _require_identifier(self.document_id, "document_id")
        content_hash = _require_identifier(self.content_hash, "content_hash")
        external_id = _require_identifier(self.external_id, "external_id")
        provider = _require_identifier(self.provider, "provider")
        canonical_source = _require_identifier(self.canonical_source, "canonical_source")
        origin_uri = _require_identifier(self.origin_uri, "origin_uri")
        source = _require_identifier(self.source, "source")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "workflow_id", workflow)
        object.__setattr__(self, "document_id", document)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "external_id", external_id)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "canonical_source", canonical_source)
        object.__setattr__(self, "origin_uri", origin_uri)
        object.__setattr__(self, "source", source)
        if self.media_type is not None:
            object.__setattr__(self, "media_type", self.media_type.strip())
        if self.title is not None:
            object.__setattr__(self, "title", self.title.strip())
        if self.language is not None:
            object.__setattr__(self, "language", self.language.strip())
        object.__setattr__(self, "tags", tuple(self.tags))
        if not isinstance(self.parser_stats, Mapping):
            raise TypeError("parser_stats_must_be_mapping")
        if not isinstance(self.provider_tags, Mapping):
            raise TypeError("provider_tags_must_be_mapping")
        object.__setattr__(self, "parser_stats", MappingProxyType(dict(self.parser_stats)))
        object.__setattr__(self, "provider_tags", MappingProxyType(dict(self.provider_tags)))


@dataclass(frozen=True)
class IngestionDecision:
    """Outcome emitted for ingestion including payload and routing reason."""

    status: IngestionStatus
    reason: str
    payload: Optional[IngestionPayload] = None
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    policy_events: Tuple[str, ...] = ()


def build_ingestion_decision(
    document: NormalizedDocument,
    delta: DeltaDecision,
    *,
    case_id: str,
    retire: bool = False,
    lifecycle: Optional[LifecycleDecision] = None,
) -> IngestionDecision:
    """Compose an ingestion decision from normalized document and delta information."""

    normalized_case_id = _require_identifier(case_id, "case_id")
    lifecycle_decision = _resolve_lifecycle(lifecycle, retire)

    if lifecycle_decision.should_retire:
        payload = _build_payload(document, delta.signatures.content_hash, normalized_case_id)
        return IngestionDecision(
            IngestionStatus.RETIRE,
            lifecycle_decision.reason,
            payload,
            lifecycle_decision.state,
            lifecycle_decision.policy_events,
        )

    if delta.status is DeltaStatus.UNCHANGED:
        return IngestionDecision(
            IngestionStatus.SKIP,
            delta.reason,
            None,
            lifecycle_decision.state,
            lifecycle_decision.policy_events,
        )

    if delta.status is DeltaStatus.NEAR_DUPLICATE:
        return IngestionDecision(
            IngestionStatus.SKIP,
            delta.reason,
            None,
            lifecycle_decision.state,
            lifecycle_decision.policy_events,
        )

    payload = _build_payload(document, delta.signatures.content_hash, normalized_case_id)
    return IngestionDecision(
        IngestionStatus.UPSERT,
        delta.reason,
        payload,
        lifecycle_decision.state,
        lifecycle_decision.policy_events,
    )


def _build_payload(
    document: NormalizedDocument, content_hash: str, case_id: str
) -> IngestionPayload:
    meta = document.meta
    external = document.external_ref
    return IngestionPayload(
        tenant_id=document.tenant_id,
        case_id=case_id,
        workflow_id=document.workflow_id,
        document_id=document.document_id,
        content_hash=content_hash,
        external_id=external.external_id,
        provider=external.provider,
        source="crawler",
        canonical_source=external.canonical_source,
        origin_uri=meta.origin_uri,
        media_type=meta.media_type,
        title=meta.title,
        language=meta.language,
        tags=meta.tags,
        parser_stats=meta.parser_stats,
        provider_tags=external.provider_tags,
    )


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
    payload: Optional[IngestionPayload],
    reason: str,
    error_code: Optional[str] = None,
    status_code: Optional[int] = None,
) -> CrawlerError:
    """Create a standardized ingestion failure payload for observability."""

    attributes: Dict[str, object] = {}
    if error_code:
        attributes["error_code"] = error_code

    source = payload.canonical_source if payload else None
    provider = payload.provider if payload else None

    return CrawlerError(
        error_class=ErrorClass.INGESTION_FAILURE,
        reason=reason,
        source=source,
        provider=provider,
        status_code=status_code,
        attributes=attributes,
    )
