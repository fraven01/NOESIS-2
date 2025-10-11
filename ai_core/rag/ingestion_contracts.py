"""Contracts and validation helpers for ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from common.logging import get_log_context, get_logger

from ai_core.infra import tracing
from ai_core.tools import InputError

from pydantic import BaseModel, ConfigDict

from .schemas import Chunk

from .vector_space_resolver import (
    VectorSpaceResolution,
    VectorSpaceResolverError,
    VectorSpaceResolverErrorCode,
    resolve_vector_space_full,
)


logger = get_logger(__name__)


class IngestionContractErrorCode:
    """Machine-readable ingestion contract error codes."""

    PROFILE_REQUIRED = "INGEST_PROFILE_REQUIRED"
    PROFILE_INVALID = "INGEST_PROFILE_INVALID"
    PROFILE_UNKNOWN = "INGEST_PROFILE_UNKNOWN"
    VECTOR_SPACE_UNKNOWN = "INGEST_VECTOR_SPACE_UNKNOWN"
    VECTOR_DIMENSION_MISMATCH = "INGEST_VECTOR_DIMENSION_MISMATCH"


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
    vector_space_id: str | None = None
    process: str | None = None
    doc_class: str | None = None

    model_config = ConfigDict(extra="forbid")


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
        tracing.emit_span(
            trace_id=trace_id,
            node_name="rag.ingestion.profile.resolve",
            metadata=metadata,
        )

    return IngestionProfileResolution(profile_id=profile_key, resolution=resolution)


def ensure_embedding_dimensions(
    chunks: Iterable[Chunk],
    expected_dimension: int | None,
    *,
    tenant_id: str | None = None,
    process: str | None = None,
    doc_class: str | None = None,
    embedding_profile: str | None = None,
    vector_space_id: str | None = None,
) -> None:
    """Raise when embeddings do not match the configured vector space dimension."""

    if expected_dimension is None:
        return

    for index, chunk in enumerate(chunks):
        embedding = chunk.embedding
        if embedding is None:
            continue
        observed = len(embedding)
        if observed != expected_dimension:
            context = {
                "tenant": tenant_id,
                "process": process,
                "doc_class": doc_class,
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


__all__ = [
    "IngestionContractErrorCode",
    "map_ingestion_error_to_status",
    "IngestionProfileResolution",
    "ChunkMeta",
    "resolve_ingestion_profile",
    "ensure_embedding_dimensions",
]
