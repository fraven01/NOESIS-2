"""Resolve vector space metadata for embedding profiles."""

from __future__ import annotations

from dataclasses import dataclass

from common.logging import get_log_context, get_logger

from ai_core.infra import tracing

from .embedding_config import (
    EmbeddingConfiguration,
    EmbeddingProfileConfig,
    VectorSpaceConfig,
    get_embedding_configuration,
)

_DOC_HINT = "See README.md (Fehlercodes Abschnitt) for remediation guidance."


logger = get_logger(__name__)


class VectorSpaceResolverError(Exception):
    """Raised when vector space resolution fails."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}. {_DOC_HINT}")
        self.code = code
        self.message = message


class VectorSpaceResolverErrorCode:
    """Machine-readable error codes for vector space resolution."""

    PROFILE_REQUIRED = "SPACE_PROFILE_REQUIRED"
    PROFILE_UNKNOWN = "SPACE_PROFILE_UNKNOWN"
    VECTOR_SPACE_UNKNOWN = "SPACE_UNDEFINED_FOR_PROFILE"


@dataclass(frozen=True, slots=True)
class VectorSpaceResolution:
    """Return type describing the resolved vector space and embedding profile."""

    profile: EmbeddingProfileConfig
    vector_space: VectorSpaceConfig


def resolve_vector_space_full(profile_id: str) -> VectorSpaceResolution:
    """Return the embedding profile and configured vector space for ``profile_id``."""

    profile_key = "" if profile_id is None else str(profile_id).strip()
    if not profile_key:
        raise VectorSpaceResolverError(
            VectorSpaceResolverErrorCode.PROFILE_REQUIRED,
            "embedding profile identifier is required",
        )

    configuration: EmbeddingConfiguration = get_embedding_configuration()
    profile: EmbeddingProfileConfig | None = configuration.embedding_profiles.get(
        profile_key
    )
    if profile is None:
        raise VectorSpaceResolverError(
            VectorSpaceResolverErrorCode.PROFILE_UNKNOWN,
            f"embedding profile '{profile_key}' is not configured",
        )

    space: VectorSpaceConfig | None = configuration.vector_spaces.get(
        profile.vector_space
    )
    if space is None:
        message = (
            "embedding profile "
            f"'{profile_key}' references missing vector space "
            f"'{profile.vector_space}'"
        )
        raise VectorSpaceResolverError(
            VectorSpaceResolverErrorCode.VECTOR_SPACE_UNKNOWN,
            message,
        )

    resolution = VectorSpaceResolution(profile=profile, vector_space=space)
    _emit_vector_space_resolution(resolution)
    return resolution


def resolve_vector_space(profile_id: str) -> VectorSpaceConfig:
    """Return the configured vector space for ``profile_id``."""

    return resolve_vector_space_full(profile_id).vector_space


def _emit_vector_space_resolution(resolution: VectorSpaceResolution) -> None:
    """Emit trace metadata describing the resolved vector space."""

    metadata = {
        "embedding_profile": resolution.profile.id,
        "vector_space_id": resolution.vector_space.id,
        "vector_space_schema": resolution.vector_space.schema,
        "vector_space_dimension": resolution.vector_space.dimension,
    }
    logger.debug("rag.vector_space.resolve", extra=metadata)
    log_context = get_log_context()
    trace_id = log_context.get("trace_id")
    if trace_id:
        tracing.emit_span(
            trace_id=trace_id,
            node_name="rag.vector_space.resolve",
            metadata=metadata,
        )


__all__ = [
    "VectorSpaceResolverError",
    "VectorSpaceResolverErrorCode",
    "VectorSpaceResolution",
    "resolve_vector_space_full",
    "resolve_vector_space",
]

