"""Pure embedding profile resolver for routing inputs."""

from __future__ import annotations

from common.logging import get_log_context, get_logger

from ai_core.infra import tracing

from .embedding_config import get_embedding_configuration
from .routing_rules import get_routing_table
from .selector_utils import normalise_selector_value


logger = get_logger(__name__)


class ProfileResolverError(Exception):
    """Raised when embedding profile resolution cannot complete."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ProfileResolverErrorCode:
    """Machine-readable error codes for profile resolver failures."""

    TENANT_REQUIRED = "RESOLVE_TENANT_REQUIRED"
    UNKNOWN_PROFILE = "RESOLVE_PROFILE_UNKNOWN"


def _normalise_optional(value: str | None) -> str | None:
    return normalise_selector_value(value)


def resolve_embedding_profile(
    *,
    tenant_id: str,
    process: str | None = None,
    doc_class: str | None = None,
) -> str:
    """Return the embedding profile identifier for the provided context."""

    tenant = tenant_id.strip()
    if not tenant:
        raise ProfileResolverError(
            ProfileResolverErrorCode.TENANT_REQUIRED,
            "tenant_id is required for embedding profile resolution",
        )

    sanitized_process = _normalise_optional(process)
    sanitized_doc_class = _normalise_optional(doc_class)
    profile_id = get_routing_table().resolve(
        tenant=tenant,
        process=sanitized_process,
        doc_class=sanitized_doc_class,
    )

    configuration = get_embedding_configuration().embedding_profiles
    if profile_id not in configuration:
        raise ProfileResolverError(
            ProfileResolverErrorCode.UNKNOWN_PROFILE,
            f"Resolved profile '{profile_id}' is not configured",
        )

    _emit_profile_resolution(
        tenant_id=tenant,
        process=sanitized_process,
        doc_class=sanitized_doc_class,
        profile_id=profile_id,
    )

    return profile_id


def _emit_profile_resolution(
    *,
    tenant_id: str,
    process: str | None,
    doc_class: str | None,
    profile_id: str,
) -> None:
    """Emit trace metadata for successful profile resolution."""

    metadata = {
        "tenant": tenant_id,
        "process": process,
        "doc_class": doc_class,
        "embedding_profile": profile_id,
    }
    logger.debug("rag.profile.resolve", extra=metadata)
    log_context = get_log_context()
    trace_id = log_context.get("trace_id")
    if trace_id:
        tracing.emit_span(
            trace_id=trace_id,
            node_name="rag.profile.resolve",
            metadata=metadata,
        )


__all__ = [
    "ProfileResolverError",
    "ProfileResolverErrorCode",
    "resolve_embedding_profile",
]
