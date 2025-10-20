"""Pure embedding profile resolver for routing inputs."""

from __future__ import annotations

from common.logging import get_log_context, get_logger

from ai_core.infra import tracing

from .embedding_config import get_embedding_configuration
from .routing_rules import get_routing_table, is_collection_routing_enabled
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
    collection_id: str | None = None,
    workflow_id: str | None = None,
    language: str | None = None,
    size: str | None = None,
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
    sanitized_collection_id = _normalise_optional(collection_id)
    sanitized_workflow_id = _normalise_optional(workflow_id)
    sanitized_language = _normalise_optional(language)
    sanitized_size = _normalise_optional(size)

    if sanitized_collection_id is None and sanitized_doc_class is not None:
        if is_collection_routing_enabled():
            sanitized_collection_id = sanitized_doc_class

    table = get_routing_table()
    resolution = table.resolve_with_metadata(
        tenant=tenant,
        process=sanitized_process,
        collection_id=sanitized_collection_id,
        workflow_id=sanitized_workflow_id,
        doc_class=sanitized_doc_class,
    )

    configuration = get_embedding_configuration().embedding_profiles
    if resolution.profile not in configuration:
        raise ProfileResolverError(
            ProfileResolverErrorCode.UNKNOWN_PROFILE,
            f"Resolved profile '{resolution.profile}' is not configured",
        )

    _emit_profile_resolution(
        tenant_id=tenant,
        process=sanitized_process,
        doc_class=sanitized_doc_class,
        collection_id=sanitized_collection_id,
        workflow_id=sanitized_workflow_id,
        language=sanitized_language,
        size=sanitized_size,
        profile_id=resolution.profile,
        resolver_path=resolution.resolver_path,
        fallback_used=resolution.fallback_used,
    )

    return resolution.profile


def _emit_profile_resolution(
    *,
    tenant_id: str,
    process: str | None,
    doc_class: str | None,
    collection_id: str | None,
    workflow_id: str | None,
    language: str | None,
    size: str | None,
    profile_id: str,
    resolver_path: str,
    fallback_used: bool,
) -> None:
    """Emit trace metadata for successful profile resolution."""

    metadata = {
        "tenant_id": tenant_id,
        "process": process,
        "doc_class": doc_class,
        "embedding_profile": profile_id,
        "collection_id": collection_id,
        "workflow_id": workflow_id,
        "language": language,
        "size": size,
        "resolver_path": resolver_path,
        "chosen_profile": profile_id,
        "fallback_used": fallback_used,
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
