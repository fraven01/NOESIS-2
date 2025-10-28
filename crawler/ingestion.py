"""Ingestion payload planning based on normalized documents and delta decisions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Protocol, Tuple

from documents.contracts import NormalizedDocument

from .contracts import Decision
from .delta import DeltaDecision
from .errors import CrawlerError, ErrorClass
from .normalizer import document_parser_stats, resolve_provider_reference
from .retire import LifecycleDecision, LifecycleState

from django.conf import settings

from ai_core.rag.ingestion_contracts import (
    ChunkMeta,
    CrawlerIngestionPayload,
    IngestionAction,
    IngestionProfileResolution,
    resolve_ingestion_profile,
)

IngestionStatus = IngestionAction


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

    if lifecycle_decision.should_retire:
        chunk_meta, profile_binding = _build_chunk_meta(
            document,
            delta.signatures.content_hash,
            normalized_case_id,
            embedding_profile,
            resolve_profile=False,
            lifecycle_state=lifecycle_decision.state,
        )
        payload = CrawlerIngestionPayload(
            action=IngestionAction.RETIRE,
            lifecycle_state=lifecycle_decision.state.value,
            policy_events=tuple(lifecycle_decision.policy_events),
            adapter_metadata=adapter_metadata,
            document_id=str(document.ref.document_id),
            workflow_id=document.ref.workflow_id,
            tenant_id=document.ref.tenant_id,
            case_id=normalized_case_id,
            content_hash=delta.signatures.content_hash,
            chunk_meta=chunk_meta,
            embedding_profile=(
                profile_binding.profile_id
                if profile_binding is not None
                else chunk_meta.embedding_profile
            ),
            vector_space_id=(
                profile_binding.resolution.vector_space.id
                if profile_binding is not None
                else chunk_meta.vector_space_id
            ),
            delta_status=delta.status.value,
        )
        return Decision(
            payload.action.value,
            lifecycle_decision.reason,
            payload.as_mapping(),
        )

    chunk_meta, profile_binding = _build_chunk_meta(
        document,
        delta.signatures.content_hash,
        normalized_case_id,
        embedding_profile,
        lifecycle_state=lifecycle_decision.state,
    )

    payload = CrawlerIngestionPayload(
        action=IngestionAction.UPSERT,
        lifecycle_state=lifecycle_decision.state.value,
        policy_events=tuple(lifecycle_decision.policy_events),
        adapter_metadata=adapter_metadata,
        document_id=str(document.ref.document_id),
        workflow_id=document.ref.workflow_id,
        tenant_id=document.ref.tenant_id,
        case_id=normalized_case_id,
        content_hash=delta.signatures.content_hash,
        chunk_meta=chunk_meta,
        embedding_profile=(
            profile_binding.profile_id
            if profile_binding is not None
            else chunk_meta.embedding_profile
        ),
        vector_space_id=(
            profile_binding.resolution.vector_space.id
            if profile_binding is not None
            else chunk_meta.vector_space_id
        ),
        delta_status=delta.status.value,
    )

    reason = (
        lifecycle_decision.reason if lifecycle_decision.should_retire else delta.reason
    )
    return Decision(payload.action.value, reason, payload.as_mapping())


def _build_chunk_meta(
    document: NormalizedDocument,
    content_hash: str,
    case_id: str,
    embedding_profile: Optional[str],
    *,
    resolve_profile: bool = True,
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
) -> Tuple[ChunkMeta, Optional[IngestionProfileResolution]]:
    profile_binding: Optional[IngestionProfileResolution]
    profile_id: Optional[str]
    vector_space_id: Optional[str]

    if resolve_profile:
        profile_binding = _resolve_profile(embedding_profile)
        profile_id = profile_binding.profile_id
        vector_space_id = profile_binding.resolution.vector_space.id
    else:
        profile_binding = None
        profile_id = str(embedding_profile) if embedding_profile else None
        vector_space_id = None
    external = resolve_provider_reference(document)
    chunk_meta = ChunkMeta(
        tenant_id=document.ref.tenant_id,
        case_id=case_id,
        source="crawler",
        hash=document.checksum,
        external_id=external.external_id,
        content_hash=content_hash,
        embedding_profile=profile_id,
        vector_space_id=vector_space_id,
        process="crawler",
        workflow_id=document.ref.workflow_id,
        collection_id=(
            str(document.ref.collection_id)
            if document.ref.collection_id is not None
            else None
        ),
        document_id=str(document.ref.document_id),
        lifecycle_state=lifecycle_state.value,
    )
    return chunk_meta, profile_binding


def _resolve_profile(
    embedding_profile: Optional[str],
) -> IngestionProfileResolution:
    if embedding_profile is None:
        default_profile = getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard")
        embedding_profile = str(default_profile)
    return resolve_ingestion_profile(embedding_profile)


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
