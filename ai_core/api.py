"""Service helpers bridging orchestration graphs with domain logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from crawler.errors import CrawlerError, ErrorClass

from ai_core.contracts.payloads import (
    CompletionPayload,
    DeltaPayload,
    EmbeddingPayload,
    GuardrailPayload as GuardrailStatePayload,
)
from ai_core.middleware import guardrails as guardrails_middleware
from ai_core.rag.guardrails import (
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)
from ai_core.rag.ingestion_contracts import ChunkMeta, resolve_ingestion_profile
from ai_core.rag.embedding_config import build_embedding_model_version
from django.utils import timezone
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_client import PgVectorClient, get_default_client
from ai_core.rag.delta import DeltaDecision, evaluate_delta

from documents.api import NormalizedDocumentPayload

from common.logging import get_log_context, get_logger

GuardrailDecision = guardrails_middleware.GuardrailDecision
GuardrailErrorCategory = guardrails_middleware.GuardrailErrorCategory


logger = get_logger(__name__)


def _build_quota_limits(config: Mapping[str, Any] | None) -> Optional[QuotaLimits]:
    if not config:
        return None
    max_documents = config.get("max_documents")
    max_bytes = config.get("max_bytes")
    if max_documents is None and max_bytes is None:
        return None
    return QuotaLimits(
        max_documents=int(max_documents) if max_documents is not None else None,
        max_bytes=int(max_bytes) if max_bytes is not None else None,
    )


def _build_quota_usage(state: Mapping[str, Any] | None) -> Optional[QuotaUsage]:
    if not state:
        return None
    documents = state.get("documents")
    bytes_used = state.get("bytes")
    if documents is None and bytes_used is None:
        return None
    return QuotaUsage(
        documents=int(documents or 0),
        bytes=int(bytes_used or 0),
    )


def _build_guardrail_limits(config: Mapping[str, Any] | None) -> GuardrailLimits:
    config = config or {}
    processing_limit = config.get("processing_time_limit")
    if isinstance(processing_limit, (int, float)) and processing_limit > 0:
        processing_timedelta = timedelta(milliseconds=float(processing_limit))
    elif isinstance(processing_limit, timedelta):
        processing_timedelta = processing_limit
    else:
        processing_timedelta = None

    mime_blacklist: Sequence[str] = tuple(config.get("mime_blacklist") or ())
    host_blocklist: Sequence[str] = tuple(config.get("host_blocklist") or ())

    return GuardrailLimits(
        max_document_bytes=(
            int(config["max_document_bytes"])
            if "max_document_bytes" in config
            and config["max_document_bytes"] is not None
            else None
        ),
        processing_time_limit=processing_timedelta,
        mime_blacklist=frozenset(
            str(entry).strip().lower() for entry in mime_blacklist if entry
        ),
        host_blocklist=frozenset(
            str(entry).strip().lower() for entry in host_blocklist if entry
        ),
        tenant_quota=_build_quota_limits(config.get("tenant_quota")),
        host_quota=_build_quota_limits(config.get("host_quota")),
    )


def _build_guardrail_signals(
    payload: NormalizedDocumentPayload,
    config: Mapping[str, Any] | None,
) -> GuardrailSignals:
    config = config or {}
    document = payload.document
    meta = document.meta
    origin_uri = meta.origin_uri or config.get("origin_uri")
    parsed_origin = urlparse(origin_uri) if origin_uri else None
    host = parsed_origin.hostname if parsed_origin else None

    tenant_usage = _build_quota_usage(config.get("tenant_usage"))
    host_usage = _build_quota_usage(config.get("host_usage"))

    mime_type = getattr(document.blob, "media_type", None)

    return GuardrailSignals(
        tenant_id=meta.tenant_id,
        provider=(meta.external_ref or {}).get("provider"),
        canonical_source=origin_uri,
        host=host,
        document_bytes=len(payload.payload_bytes),
        mime_type=mime_type,
        tenant_usage=tenant_usage,
        host_usage=host_usage,
    )


def enforce_guardrails(
    *,
    normalized_document: NormalizedDocumentPayload,
    config: Optional[Mapping[str, Any]] = None,
    limits: GuardrailLimits | None = None,
    signals: GuardrailSignals | None = None,
    error_builder: Optional[guardrails_middleware.ErrorBuilder] = None,
    frontier_state: Optional[Mapping[str, Any]] = None,
) -> GuardrailDecision:
    """Apply guardrail evaluation using the shared middleware implementation."""

    limits = limits or _build_guardrail_limits(config)
    signals = signals or _build_guardrail_signals(normalized_document, config)
    builder = error_builder or _build_guardrail_error
    decision = guardrails_middleware.enforce_guardrails(
        limits=limits,
        signals=signals,
        error_builder=builder,
    )

    frontier_payload = _project_frontier_state(frontier_state)
    attributes: Dict[str, Any] = dict(decision.attributes)
    if frontier_payload:
        attributes.setdefault("frontier", frontier_payload)

    if frontier_payload:
        frontier_events = tuple(frontier_payload.get("policy_events", ()))
    else:
        frontier_events = ()
    merged_events = _merge_policy_events(decision.policy_events, frontier_events)
    if merged_events:
        attributes["policy_events"] = merged_events

    banned_terms = tuple(
        str(term).lower() for term in (config or {}).get("banned_terms", ())
    )
    if decision.allowed and banned_terms:
        text = normalized_document.primary_text.lower()
        for term in banned_terms:
            if term and term in text:
                denied = GuardrailDecision(
                    "deny",
                    "term_blocked",
                    {"policy_events": ("term_blocked",), "term": term},
                )
                if frontier_payload:
                    attrs = dict(denied.attributes)
                    attrs["frontier"] = frontier_payload
                    attrs["policy_events"] = _merge_policy_events(
                        denied.policy_events, frontier_events
                    )
                    return GuardrailDecision(denied.decision, denied.reason, attrs)
                return denied

    return GuardrailDecision(decision.decision, decision.reason, attributes)


def _build_guardrail_error(
    category: GuardrailErrorCategory,
    reason: str,
    signals: GuardrailSignals,
    attributes: Mapping[str, Any],
) -> CrawlerError:
    error_class = {
        GuardrailErrorCategory.POLICY_DENY: ErrorClass.POLICY_DENY,
        GuardrailErrorCategory.TIMEOUT: ErrorClass.TIMEOUT,
    }.get(category, ErrorClass.POLICY_DENY)
    return CrawlerError(
        error_class=error_class,
        reason=reason,
        source=signals.canonical_source,
        provider=signals.provider,
        attributes=dict(attributes or {}),
    )


def decide_delta(
    *,
    normalized_document: NormalizedDocumentPayload,
    baseline: Optional[Mapping[str, Any]] = None,
    frontier_state: Optional[Mapping[str, Any]] = None,
) -> DeltaDecision:
    """Evaluate delta against previous signatures using crawler heuristics."""

    baseline = baseline or {}
    previous_hash = baseline.get("content_hash") or baseline.get("checksum")
    baseline_primary_text = baseline.get("primary_text")
    baseline_payload = baseline.get("payload_bytes")
    previous_version = baseline.get("version")
    if previous_version is not None:
        try:
            previous_version = int(previous_version)
        except (TypeError, ValueError):
            previous_version = None

    primary_text = baseline_primary_text
    if not isinstance(primary_text, str):
        primary_text = normalized_document.primary_text

    binary_payload = baseline_payload
    if not isinstance(binary_payload, (bytes, bytearray)):
        binary_payload = normalized_document.payload_bytes

    decision = evaluate_delta(
        normalized_document.document,
        primary_text=primary_text,
        previous_content_hash=previous_hash,
        previous_version=previous_version,
        binary_payload=bytes(binary_payload),
    )

    frontier_payload = _project_frontier_state(frontier_state)
    if frontier_payload:
        attributes: Dict[str, Any] = dict(decision.attributes)
        attributes.setdefault("frontier", frontier_payload)
        policy_events = tuple(frontier_payload.get("policy_events", ()))
        if policy_events:
            attributes["policy_events"] = _merge_policy_events((), policy_events)
        decision = DeltaDecision(decision.decision, decision.reason, attributes)

    changed_fields: list[str] = []
    if isinstance(baseline_primary_text, str) and (
        baseline_primary_text != normalized_document.primary_text
    ):
        changed_fields.append("primary_text")
    if isinstance(baseline_payload, (bytes, bytearray)) and (
        bytes(baseline_payload) != normalized_document.payload_bytes
    ):
        changed_fields.append("payload_bytes")

    signatures = decision.attributes.get("signatures")
    content_hash = None
    if signatures is not None:
        content_hash = getattr(signatures, "content_hash", None)
    if previous_hash and content_hash and previous_hash != content_hash:
        changed_fields.append("content_hash")

    if (
        previous_version is not None
        and decision.version is not None
        and decision.version != previous_version
    ):
        changed_fields.append("version")

    log_context = get_log_context()
    metadata = normalized_document.metadata
    document = normalized_document.document
    log_payload: Dict[str, Any] = {
        "decision": decision.decision,
        "reason": decision.reason,
        "changed_fields": tuple(changed_fields),
        "tenant_id": document.ref.tenant_id,
        "document_id": normalized_document.document_id,
        "workflow_id": document.ref.workflow_id,
        "case_id": metadata.get("case_id"),
        "origin_uri": metadata.get("origin_uri"),
        "provider": metadata.get("provider"),
        "source": metadata.get("source") or document.source,
        "content_hash": content_hash,
        "baseline_content_hash": previous_hash,
        "version": decision.version,
        "baseline_version": previous_version,
    }

    if signatures is not None and getattr(signatures, "near_duplicate", None):
        near_duplicate = getattr(signatures, "near_duplicate", None)
        if getattr(near_duplicate, "fingerprint", None):
            log_payload["near_duplicate_fingerprint"] = near_duplicate.fingerprint

    policy_events = decision.attributes.get("policy_events")
    if policy_events:
        log_payload["policy_events"] = tuple(policy_events)

    trace_id = log_context.get("trace_id") or metadata.get("trace_id")
    span_id = log_context.get("span_id")
    if trace_id:
        log_payload["trace_id"] = trace_id
    if span_id:
        log_payload["span_id"] = span_id

    context_case = log_context.get("case_id")
    if context_case and not log_payload.get("case_id"):
        log_payload["case_id"] = context_case
    context_tenant = log_context.get("tenant") or log_context.get("tenant_id")
    if context_tenant:
        log_payload.setdefault("tenant_id", context_tenant)

    if frontier_payload:
        log_payload["frontier"] = frontier_payload

    filtered_payload = {
        key: value for key, value in log_payload.items() if value is not None
    }
    filtered_payload["changed_fields"] = tuple(changed_fields)

    logger.info("crawler.decide_delta", extra=filtered_payload)

    return decision


def _merge_policy_events(
    existing: Iterable[str], extras: Iterable[str]
) -> Tuple[str, ...]:
    seen: Dict[str, None] = {}
    for event in (*existing, *extras):
        if not event:
            continue
        key = str(event).strip()
        if key:
            seen.setdefault(key, None)
    return tuple(seen.keys())


def _project_frontier_state(
    frontier_state: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(frontier_state, Mapping):
        return {}

    def _serialise(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            projected: Dict[str, Any] = {}
            for key, inner in value.items():
                serialised = _serialise(inner)
                if serialised is None:
                    continue
                cleaned = serialised
                if cleaned in ((), {}, None):
                    continue
                projected[str(key)] = cleaned
            return projected
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            items = []
            for entry in value:
                serialised = _serialise(entry)
                if serialised is None:
                    continue
                if serialised in ((), {}, None):
                    continue
                items.append(serialised)
            return tuple(items)
        return str(value)

    projected: Dict[str, Any] = {}
    for key, value in frontier_state.items():
        serialised = _serialise(value)
        if serialised is None:
            continue
        if serialised in ((), {}, None):
            continue
        projected[str(key)] = serialised
    return projected


@dataclass(frozen=True)
class EmbeddingResult:
    """Outcome of the embedding trigger step."""

    status: str
    chunks_inserted: int
    embedding_profile: Optional[str]
    vector_space_id: Optional[str]
    chunk_meta: ChunkMeta

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "status": self.status,
            "chunks_inserted": self.chunks_inserted,
            "embedding_profile": self.embedding_profile,
            "vector_space_id": self.vector_space_id,
            "chunk_meta": self.chunk_meta.model_dump(),
        }


def _resolve_vector_client(
    factory: Optional[Callable[[], PgVectorClient]] = None,
) -> PgVectorClient:
    if factory is None:
        return get_default_client()
    client = factory()
    if not isinstance(client, PgVectorClient):
        raise TypeError("vector_client_factory_invalid")
    return client


def trigger_embedding(
    *,
    normalized_document: NormalizedDocumentPayload,
    embedding_profile: Optional[str] = None,
    tenant_id: Optional[str] = None,
    case_id: Optional[str] = None,
    vector_client: Optional[PgVectorClient] = None,
    vector_client_factory: Optional[Callable[[], PgVectorClient]] = None,
    chunks: Optional[Sequence[Mapping[str, Any]]] = None,
    chunker: Optional[str] = None,
    chunker_mode: Optional[str] = None,
    context: Any = None,
    config: Any = None,
) -> EmbeddingResult:
    """Persist chunk metadata and forward the document to the vector client."""

    document = normalized_document.document
    tenant = tenant_id or document.ref.tenant_id

    case = (
        case_id
        or normalized_document.metadata.get("case_id")
        or getattr(getattr(context, "metadata", None), "case_id", None)
        or "default"
    )

    profile_key = (
        embedding_profile or getattr(config, "embedding_profile", None) or "standard"
    )
    profile_resolution = resolve_ingestion_profile(profile_key)
    embedding_model_version = build_embedding_model_version(
        profile_resolution.resolution.profile
    )
    embedding_created_at = timezone.now().isoformat()

    chunk_meta = ChunkMeta(
        tenant_id=tenant,
        case_id=str(case),
        source=document.source or "crawler",
        hash=document.checksum,
        external_id=(document.meta.external_ref or {}).get(
            "external_id", str(document.ref.document_id)
        ),
        content_hash=document.checksum,
        embedding_profile=profile_resolution.profile_id,
        embedding_model_version=embedding_model_version,
        embedding_created_at=embedding_created_at,
        vector_space_id=profile_resolution.resolution.vector_space.id,
        workflow_id=document.ref.workflow_id,
        document_id=str(document.ref.document_id),
        document_version_id=(
            str(document.ref.document_version_id)
            if getattr(document.ref, "document_version_id", None) is not None
            else None
        ),
        is_latest=True,
        lifecycle_state=document.lifecycle_state,
        chunker=chunker,
        chunker_mode=chunker_mode,
    )

    base_meta = chunk_meta.model_dump(exclude_none=True)
    meta_trace = getattr(getattr(context, "metadata", None), "trace_id", None)
    if meta_trace:
        base_meta["trace_id"] = meta_trace

    chunk_payloads: list[Chunk] = []
    chunk_texts: list[str] = []  # For batch embedding

    if chunks:
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            text_value = chunk.get("text") or chunk.get("content") or ""
            chunk_content = str(text_value) if text_value is not None else ""
            if not chunk_content:
                continue

            meta_payload = dict(base_meta)
            # Merge chunker-provided metadata if present
            chunk_specific_meta = chunk.get("metadata")
            if isinstance(chunk_specific_meta, Mapping):
                meta_payload.update(chunk_specific_meta)

            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                meta_payload["chunk_id"] = str(chunk_id)

            parent_ref = chunk.get("parent_ref")
            if parent_ref:
                meta_payload["parent_ref"] = parent_ref

            parent_ids = chunk.get("parent_ids")
            if parent_ids:
                meta_payload["parent_ids"] = list(parent_ids)
            elif parent_ref:
                meta_payload["parent_ids"] = [parent_ref]

            section_path = chunk.get("section_path")
            if section_path:
                meta_payload["section_path"] = list(section_path)

            page_index = chunk.get("page_index")
            if page_index is not None:
                meta_payload["page_index"] = page_index

            kind_value = chunk.get("kind")
            if kind_value:
                meta_payload["kind"] = kind_value

            index_value = chunk.get("index")
            if index_value is not None:
                meta_payload["index"] = index_value

            chunk_payloads.append(
                Chunk(
                    content=chunk_content,
                    meta=meta_payload,
                    embedding=None,  # Will be set after batch embedding
                )
            )
            chunk_texts.append(chunk_content)
    else:
        normalized_content = (
            normalized_document.content_normalized
            or normalized_document.primary_text
            or ""
        )

        chunk_payloads.append(
            Chunk(
                content=normalized_content,
                meta={
                    "tenant_id": chunk_meta.tenant_id,
                    "case_id": chunk_meta.case_id,
                    "source": chunk_meta.source,
                    "hash": chunk_meta.hash,
                    "external_id": chunk_meta.external_id,
                    "content_hash": chunk_meta.content_hash,
                    "embedding_profile": chunk_meta.embedding_profile,
                    "embedding_model_version": chunk_meta.embedding_model_version,
                    "embedding_created_at": chunk_meta.embedding_created_at,
                    "vector_space_id": chunk_meta.vector_space_id,
                    "workflow_id": chunk_meta.workflow_id,
                    "document_id": chunk_meta.document_id,
                    "lifecycle_state": chunk_meta.lifecycle_state,
                },
                embedding=None,  # Will be set after batch embedding
            )
        )
        chunk_texts.append(normalized_content)

    # ✅ CRITICAL FIX: Calculate embeddings using EmbeddingClient
    if chunk_texts:
        from ai_core.rag.embeddings import EmbeddingClient

        embedding_client = EmbeddingClient.from_settings()
        embedding_result = embedding_client.embed(chunk_texts)

        # Assign embeddings to chunks
        for i, chunk in enumerate(chunk_payloads):
            if i < len(embedding_result.vectors):
                chunk_payloads[i] = chunk.model_copy(
                    update={"embedding": embedding_result.vectors[i]}
                )

    client = vector_client or _resolve_vector_client(vector_client_factory)
    inserted = client.upsert_chunks(chunk_payloads)

    return EmbeddingResult(
        status="upserted" if inserted else "skipped",
        chunks_inserted=inserted,
        embedding_profile=chunk_meta.embedding_profile,
        vector_space_id=chunk_meta.vector_space_id,
        chunk_meta=chunk_meta,
    )


def build_completion_payload(
    *,
    normalized_document: NormalizedDocumentPayload,
    decision: DeltaDecision,
    guardrails: GuardrailDecision,
    embedding_result: EmbeddingResult | None,
) -> CompletionPayload:
    """Assemble a stable payload describing the graph summary."""

    document_payload = normalized_document.document.model_dump()
    document_payload["checksum"] = normalized_document.checksum

    attributes = dict(getattr(decision, "attributes", {}))
    delta_payload = DeltaPayload(
        decision=decision.decision,
        reason=decision.reason,
        content_hash=attributes.get("content_hash")
        or getattr(decision, "content_hash", None),
        version=getattr(decision, "version", None),
        changed_fields=tuple(attributes.get("changed_fields", ())),
        policy_events=tuple(attributes.get("policy_events", ())),
        attributes=attributes,
    )

    guardrail_attributes = dict(getattr(guardrails, "attributes", {}))
    guardrail_payload = GuardrailStatePayload(
        decision=guardrails.decision,
        reason=guardrails.reason,
        allowed=getattr(guardrails, "allowed", False),
        policy_events=tuple(getattr(guardrails, "policy_events", ())),
        limits=None,
        signals=None,
        attributes=guardrail_attributes,
    )

    embedding_payload: EmbeddingPayload | None = None
    if embedding_result is not None:
        chunk_meta = embedding_result.chunk_meta
        embedding_payload = EmbeddingPayload(
            status=embedding_result.status,
            chunks_inserted=embedding_result.chunks_inserted,
            embedding_profile=embedding_result.embedding_profile or "",
            vector_space_id=embedding_result.vector_space_id or "",
            tenant_id=chunk_meta.tenant_id,
            case_id=chunk_meta.case_id,
            document_id=chunk_meta.document_id,
            workflow_id=chunk_meta.workflow_id,
            collection_id=chunk_meta.collection_id,
        )

    return CompletionPayload(
        normalized_document=document_payload,
        delta=delta_payload,
        guardrails=guardrail_payload,
        embedding=embedding_payload,
    )


__all__ = [
    "EmbeddingResult",
    "GuardrailDecision",
    "DeltaDecision",
    "build_completion_payload",
    "decide_delta",
    "enforce_guardrails",
    "trigger_embedding",
]
