"""Contracts and validation helpers for ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import random
from typing import Iterable, Mapping, MutableMapping, Sequence
from types import MappingProxyType

from django.utils import timezone

from common.logging import get_log_context, get_logger

from ai_core.infra.observability import record_span
from ai_core.tools import InputError

from pydantic import BaseModel, ConfigDict

from documents.contracts import NormalizedDocument
from documents.providers import parse_provider_reference

from ai_core.rag.vector_client import DedupSignatures

from .schemas import Chunk

from .vector_space_resolver import (
    VectorSpaceResolution,
    VectorSpaceResolverError,
    VectorSpaceResolverErrorCode,
    resolve_vector_space_full,
)
from .embedding_config import build_embedding_model_version


logger = get_logger(__name__)
_ZERO_EPSILON = 1e-12


class IngestionContractErrorCode:
    """Machine-readable ingestion contract error codes."""

    PROFILE_REQUIRED = "INGEST_PROFILE_REQUIRED"
    PROFILE_INVALID = "INGEST_PROFILE_INVALID"
    PROFILE_UNKNOWN = "INGEST_PROFILE_UNKNOWN"
    VECTOR_SPACE_UNKNOWN = "INGEST_VECTOR_SPACE_UNKNOWN"
    VECTOR_DIMENSION_MISMATCH = "INGEST_VECTOR_DIMENSION_MISMATCH"
    EMBEDDING_INVALID = "INGEST_EMBEDDING_INVALID"
    EMBEDDING_ZERO = "INGEST_EMBEDDING_ZERO"


def map_ingestion_error_to_status(code: str) -> int:
    """Return HTTP status mapped to an ingestion contract error code."""

    # All ingestion contract violations are client errors (bad request).
    return 400


@dataclass(frozen=True, slots=True)
class IngestionProfileResolution:
    """Describe the resolved embedding profile and vector space for ingestion."""

    profile_id: str
    resolution: VectorSpaceResolution


class ChunkMeta(BaseModel):
    """Validated chunk metadata persisted alongside embeddings."""

    tenant_id: str
    case_id: str
    source: str
    hash: str
    external_id: str
    content_hash: str
    embedding_profile: str | None = None
    embedding_model_version: str | None = None
    embedding_created_at: str | None = None
    vector_space_id: str | None = None
    process: str | None = None
    workflow_id: str | None = None
    parent_ids: list[str] | None = None
    collection_id: str | None = None
    document_id: str | None = None
    lifecycle_state: str | None = None
    trace_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class IngestionAction(str, Enum):
    """Supported ingestion outcomes emitted by crawler planning."""

    UPSERT = "upsert"
    SKIP = "skip"
    RETIRE = "retire"


class CrawlerIngestionPayload(BaseModel):
    """Normalized crawler ingestion metadata forwarded to vector services."""

    action: IngestionAction
    lifecycle_state: str
    policy_events: tuple[str, ...] = ()
    adapter_metadata: Mapping[str, object]
    document_id: str
    workflow_id: str | None = None
    tenant_id: str
    case_id: str
    content_hash: str | None = None
    chunk_meta: ChunkMeta | None = None
    embedding_profile: str | None = None
    vector_space_id: str | None = None
    delta_status: str | None = None
    content_raw: str | None = None
    content_normalized: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    def as_mapping(self) -> Mapping[str, object]:
        """Return an immutable view of the payload for decision attributes."""

        payload: MutableMapping[str, object] = {
            "lifecycle_state": self.lifecycle_state,
            "policy_events": tuple(self.policy_events),
            "adapter_metadata": self.adapter_metadata,
            "document_id": self.document_id,
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
            "case_id": self.case_id,
            "content_hash": self.content_hash,
            "chunk_meta": self.chunk_meta,
            "embedding_profile": self.embedding_profile,
            "vector_space_id": self.vector_space_id,
            "delta_status": self.delta_status,
            "content_raw": self.content_raw,
            "content_normalized": self.content_normalized,
        }
        return dict(payload)


def resolve_ingestion_profile(profile: object | None) -> IngestionProfileResolution:
    """Validate and resolve the embedding profile for ingestion."""

    raw_context = {"embedding_profile": profile}
    if profile is None:
        raise InputError(
            IngestionContractErrorCode.PROFILE_REQUIRED,
            "embedding_profile is required",
            context=raw_context,
        )

    if not isinstance(profile, str):
        raise InputError(
            IngestionContractErrorCode.PROFILE_INVALID,
            "embedding_profile must be a non-empty string",
            context=raw_context,
        )

    profile_key = profile.strip()
    if not profile_key:
        raise InputError(
            IngestionContractErrorCode.PROFILE_REQUIRED,
            "embedding_profile must not be empty",
            context=raw_context,
        )

    try:
        resolution = resolve_vector_space_full(profile_key)
    except VectorSpaceResolverError as exc:
        mapping = {
            VectorSpaceResolverErrorCode.PROFILE_REQUIRED: IngestionContractErrorCode.PROFILE_REQUIRED,
            VectorSpaceResolverErrorCode.PROFILE_UNKNOWN: IngestionContractErrorCode.PROFILE_UNKNOWN,
            VectorSpaceResolverErrorCode.VECTOR_SPACE_UNKNOWN: IngestionContractErrorCode.VECTOR_SPACE_UNKNOWN,
        }
        mapped_code = mapping.get(exc.code, IngestionContractErrorCode.PROFILE_UNKNOWN)
        raise InputError(
            mapped_code,
            exc.message,
            context={"embedding_profile": profile_key},
        ) from exc

    metadata = {
        "embedding_profile": profile_key,
        "vector_space_id": resolution.vector_space.id,
        "vector_space_schema": resolution.vector_space.schema,
        "vector_space_dimension": resolution.vector_space.dimension,
    }
    logger.debug("rag.ingestion.profile.resolve", extra=metadata)
    log_context = get_log_context()
    trace_id = log_context.get("trace_id")
    if trace_id:
        record_span(
            "rag.ingestion.profile.resolve",
            trace_id=str(trace_id),
            attributes=metadata,
        )

    return IngestionProfileResolution(profile_id=profile_key, resolution=resolution)


def ensure_embedding_dimensions(
    chunks: Iterable[Chunk],
    expected_dimension: int | None,
    *,
    tenant_id: str | None = None,
    process: str | None = None,
    workflow_id: str | None = None,
    embedding_profile: str | None = None,
    vector_space_id: str | None = None,
) -> None:
    """Raise when embeddings fail pre-upsert validation checks."""

    if expected_dimension is not None and expected_dimension < 0:
        expected_dimension = None

    for index, chunk in enumerate(chunks):
        embedding = chunk.embedding
        if embedding is None:
            continue
        floats = _coerce_embedding_values(embedding)
        if floats is None:
            _raise_embedding_invalid(
                "embedding contains non-numeric values",
                index=index,
                tenant_id=tenant_id,
                process=process,
                workflow_id=workflow_id,
                embedding_profile=embedding_profile,
                vector_space_id=vector_space_id,
                chunk=chunk,
            )

        invalid_index = _first_non_finite_index(floats)
        if invalid_index is not None:
            _raise_embedding_invalid(
                "embedding contains NaN or Inf values",
                index=index,
                tenant_id=tenant_id,
                process=process,
                workflow_id=workflow_id,
                embedding_profile=embedding_profile,
                vector_space_id=vector_space_id,
                chunk=chunk,
                invalid_index=invalid_index,
                invalid_value=floats[invalid_index],
                invalid_reason="non_finite",
            )

        observed = len(floats)
        if expected_dimension is not None and observed != expected_dimension:
            context = {
                "tenant": tenant_id,
                "process": process,
                "workflow_id": workflow_id,
                "embedding_profile": embedding_profile,
                "vector_space_id": vector_space_id,
                "expected_dimension": expected_dimension,
                "observed_dimension": observed,
                "chunk_index": index,
            }
            external_id = chunk.meta.get("external_id") if chunk.meta else None
            if external_id is not None:
                context["external_id"] = external_id
            raise InputError(
                IngestionContractErrorCode.VECTOR_DIMENSION_MISMATCH,
                (
                    "embedding dimension mismatch detected before persistence: "
                    f"expected {expected_dimension}, observed {observed}"
                ),
                context=context,
            )

        norm_sq = math.fsum(value * value for value in floats)
        if norm_sq <= _ZERO_EPSILON:
            context = _base_embedding_context(
                index=index,
                tenant_id=tenant_id,
                process=process,
                workflow_id=workflow_id,
                embedding_profile=embedding_profile,
                vector_space_id=vector_space_id,
                chunk=chunk,
            )
            context["norm_sq"] = norm_sq
            raise InputError(
                IngestionContractErrorCode.EMBEDDING_ZERO,
                "embedding vector norm is zero; refusing to persist",
                context=context,
            )


def _coerce_embedding_values(values: Iterable[object]) -> list[float] | None:
    try:
        floats = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return floats


def _first_non_finite_index(values: Sequence[float]) -> int | None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            return index
    return None


def _base_embedding_context(
    *,
    index: int,
    tenant_id: str | None,
    process: str | None,
    workflow_id: str | None,
    embedding_profile: str | None,
    vector_space_id: str | None,
    chunk: Chunk,
) -> dict[str, object]:
    context: dict[str, object] = {
        "tenant": tenant_id,
        "process": process,
        "workflow_id": workflow_id,
        "embedding_profile": embedding_profile,
        "vector_space_id": vector_space_id,
        "chunk_index": index,
    }
    external_id = chunk.meta.get("external_id") if chunk.meta else None
    if external_id is not None:
        context["external_id"] = external_id
    return context


def _raise_embedding_invalid(
    message: str,
    *,
    index: int,
    tenant_id: str | None,
    process: str | None,
    workflow_id: str | None,
    embedding_profile: str | None,
    vector_space_id: str | None,
    chunk: Chunk,
    invalid_index: int | None = None,
    invalid_value: float | None = None,
    invalid_reason: str | None = None,
) -> None:
    context = _base_embedding_context(
        index=index,
        tenant_id=tenant_id,
        process=process,
        workflow_id=workflow_id,
        embedding_profile=embedding_profile,
        vector_space_id=vector_space_id,
        chunk=chunk,
    )
    if invalid_index is not None:
        context["invalid_index"] = invalid_index
    if invalid_value is not None:
        context["invalid_value"] = invalid_value
    if invalid_reason:
        context["invalid_reason"] = invalid_reason
    raise InputError(
        IngestionContractErrorCode.EMBEDDING_INVALID,
        message,
        context=context,
    )


def _compute_embedding_quality_stats(
    chunks: Iterable[Chunk],
    *,
    sample_size: int,
    outlier_threshold: float,
) -> dict[str, object] | None:
    vectors: list[tuple[int, list[float], str | None]] = []

    for index, chunk in enumerate(chunks):
        embedding = chunk.embedding
        if embedding is None:
            continue
        floats = _coerce_embedding_values(embedding)
        if not floats:
            continue
        if _first_non_finite_index(floats) is not None:
            continue
        norm_sq = math.fsum(value * value for value in floats)
        if norm_sq <= _ZERO_EPSILON:
            continue
        norm = math.sqrt(norm_sq)
        if not math.isfinite(norm) or norm <= _ZERO_EPSILON:
            continue
        unit = [value / norm for value in floats]
        external_id = chunk.meta.get("external_id") if chunk.meta else None
        vectors.append((index, unit, str(external_id) if external_id else None))

    total_vectors = len(vectors)
    if total_vectors < 2:
        return None
    dimension = len(vectors[0][1])
    vectors = [entry for entry in vectors if len(entry[1]) == dimension]
    total_vectors = len(vectors)
    if total_vectors < 2:
        return None

    effective_sample = min(max(2, sample_size), total_vectors)
    trace_id = get_log_context().get("trace_id")
    seed_source = str(trace_id) if trace_id else str(total_vectors)
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    sample = rng.sample(vectors, k=effective_sample)

    ref_vector = rng.choice(sample)[1]
    mean_cosine = 0.0
    for _, vector, _ in sample:
        mean_cosine += _dot_product(ref_vector, vector)
    mean_cosine = mean_cosine / float(effective_sample)

    mean_vector = [0.0] * dimension
    for _, vector, _ in sample:
        for idx, value in enumerate(vector):
            mean_vector[idx] += value
    mean_vector = [value / float(effective_sample) for value in mean_vector]
    mean_norm_sq = math.fsum(value * value for value in mean_vector)
    if mean_norm_sq <= _ZERO_EPSILON:
        return {
            "embedding_count": total_vectors,
            "sample_size": effective_sample,
            "sample_mean_cosine": round(mean_cosine, 6),
            "outlier_threshold": outlier_threshold,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
        }

    mean_norm = math.sqrt(mean_norm_sq)
    if not math.isfinite(mean_norm) or mean_norm <= _ZERO_EPSILON:
        return None
    mean_unit = [value / mean_norm for value in mean_vector]

    outliers: list[dict[str, object]] = []
    outlier_count = 0
    for index, vector, external_id in sample:
        similarity = _dot_product(mean_unit, vector)
        if similarity < outlier_threshold:
            outlier_count += 1
            if len(outliers) < 5:
                outlier_entry = {"chunk_index": index, "similarity": similarity}
                if external_id:
                    outlier_entry["external_id"] = external_id
                outliers.append(outlier_entry)

    outlier_ratio = outlier_count / float(effective_sample)

    payload: dict[str, object] = {
        "embedding_count": total_vectors,
        "sample_size": effective_sample,
        "sample_mean_cosine": round(mean_cosine, 6),
        "outlier_threshold": outlier_threshold,
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_ratio, 6),
    }
    if outliers:
        payload["outlier_examples"] = outliers
    return payload


def _dot_product(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return math.fsum(value * vec_b[idx] for idx, value in enumerate(vec_a))


def log_embedding_quality_stats(
    chunks: Iterable[Chunk],
    *,
    tenant_id: str | None = None,
    process: str | None = None,
    workflow_id: str | None = None,
    embedding_profile: str | None = None,
    vector_space_id: str | None = None,
    sample_size: int = 256,
    outlier_threshold: float = 0.1,
) -> dict[str, object] | None:
    """Log cosine similarity stats and outlier counts for a chunk batch."""

    stats = _compute_embedding_quality_stats(
        chunks,
        sample_size=sample_size,
        outlier_threshold=outlier_threshold,
    )
    if not stats:
        return None

    payload: dict[str, object] = {
        "tenant_id": tenant_id,
        "process": process,
        "workflow_id": workflow_id,
        "embedding_profile": embedding_profile,
        "vector_space_id": vector_space_id,
        **stats,
    }

    logger.info("embedding.quality.stats", extra=payload)
    outlier_count = stats.get("outlier_count", 0)
    if isinstance(outlier_count, int) and outlier_count > 0:
        logger.warning("embedding.quality.outliers", extra=payload)

    trace_id = get_log_context().get("trace_id")
    if trace_id:
        record_span(
            "embedding.quality.stats",
            trace_id=str(trace_id),
            attributes=payload,
        )

    return payload


__all__ = [
    "IngestionContractErrorCode",
    "map_ingestion_error_to_status",
    "IngestionProfileResolution",
    "ChunkMeta",
    "IngestionAction",
    "CrawlerIngestionPayload",
    "build_crawler_ingestion_payload",
    "resolve_ingestion_profile",
    "ensure_embedding_dimensions",
    "log_embedding_quality_stats",
]


def build_crawler_ingestion_payload(
    *,
    document: NormalizedDocument,
    signatures: DedupSignatures,
    case_id: str,
    action: IngestionAction,
    lifecycle_state: object,
    policy_events: Iterable[object] = (),
    adapter_metadata: Mapping[str, object] | None = None,
    embedding_profile: str | None = None,
    delta_status: object | None = None,
    source: str | None = None,
    process: str | None = None,
) -> CrawlerIngestionPayload:
    """Compose a :class:`CrawlerIngestionPayload` from crawler planning hints."""

    normalized_case_id = _require_identifier(case_id, "case_id")
    lifecycle_value = _normalize_lifecycle_state(lifecycle_state)
    policy_tuple = _normalize_policy_events(policy_events)
    frozen_adapter_metadata = _freeze_adapter_metadata(adapter_metadata)

    profile_id: str | None
    vector_space_id: str | None
    embedding_model_version: str | None
    embedding_created_at: str | None

    if action is IngestionAction.RETIRE:
        profile_id = _coerce_optional_string(embedding_profile)
        vector_space_id = None
        embedding_model_version = None
        embedding_created_at = None
    else:
        profile_binding = _resolve_embedding_profile(embedding_profile)
        profile_id = profile_binding.profile_id
        vector_space_id = profile_binding.resolution.vector_space.id
        embedding_model_version = build_embedding_model_version(
            profile_binding.resolution.profile
        )
        embedding_created_at = timezone.now().isoformat()

    provider = parse_provider_reference(document.meta)

    resolved_source = (
        _coerce_optional_string(source)
        or _coerce_optional_string(getattr(document, "source", None))
        or "crawler"
    )
    resolved_process = _coerce_optional_string(process) or resolved_source

    chunk_meta = ChunkMeta(
        tenant_id=document.ref.tenant_id,
        case_id=normalized_case_id,
        source=resolved_source,
        hash=document.checksum,
        external_id=provider.external_id,
        content_hash=signatures.content_hash,
        embedding_profile=profile_id,
        embedding_model_version=embedding_model_version,
        embedding_created_at=embedding_created_at,
        vector_space_id=vector_space_id,
        process=resolved_process,
        workflow_id=document.ref.workflow_id,
        collection_id=(
            str(document.ref.collection_id)
            if document.ref.collection_id is not None
            else None
        ),
        document_id=str(document.ref.document_id),
        lifecycle_state=lifecycle_value,
    )

    return CrawlerIngestionPayload(
        action=action,
        lifecycle_state=lifecycle_value,
        policy_events=policy_tuple,
        adapter_metadata=frozen_adapter_metadata,
        document_id=str(document.ref.document_id),
        workflow_id=document.ref.workflow_id,
        tenant_id=document.ref.tenant_id,
        case_id=normalized_case_id,
        content_hash=signatures.content_hash,
        chunk_meta=chunk_meta,
        embedding_profile=profile_id,
        vector_space_id=vector_space_id,
        delta_status=_coerce_optional_string(delta_status),
    )


def _freeze_adapter_metadata(
    metadata: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if metadata is None:
        return {}
    if isinstance(metadata, MappingProxyType):
        return dict(metadata)
    if not isinstance(metadata, Mapping):
        return dict(metadata)
    return dict(metadata)


def _normalize_policy_events(events: Iterable[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    for event in events:
        text = _coerce_optional_string(event)
        if text:
            normalized.append(text)
    return tuple(normalized)


def _normalize_lifecycle_state(value: object) -> str:
    candidate = _coerce_optional_string(value)
    if candidate:
        return candidate
    return "active"


def _require_identifier(value: str | None, field: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError(f"{field}_required")
    return candidate


def _coerce_optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    return text or None


def _resolve_embedding_profile(
    embedding_profile: str | None,
) -> IngestionProfileResolution:
    from django.conf import settings

    if embedding_profile is None:
        default_profile = getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard")
        embedding_profile = str(default_profile)
    return resolve_ingestion_profile(embedding_profile)
