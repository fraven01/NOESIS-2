"""RAG vector store abstractions and helpers."""

from __future__ import annotations

from .profile_resolver import (
    ProfileResolverError,
    ProfileResolverErrorCode,
    resolve_embedding_profile,
)
from .limits import CandidatePoolPolicy, resolve_candidate_pool_policy
from .router_validation import (
    RouterInputError,
    RouterInputErrorCode,
    emit_router_validation_failure,
    map_router_error_to_status,
    validate_search_inputs,
)
from .guardrails import (
    FetcherLimits,
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)
from .ingestion_contracts import (
    IngestionContractErrorCode,
    ensure_embedding_dimensions,
    map_ingestion_error_to_status,
    resolve_ingestion_profile,
)
from .vector_store import (
    TenantScopedVectorStore,
    VectorStore,
    VectorStoreRouter,
    get_default_router,
)
from .vector_space_resolver import (
    VectorSpaceResolverError,
    VectorSpaceResolverErrorCode,
    resolve_vector_space_full,
    resolve_vector_space,
)
from .vector_schema import (
    VectorSchemaError,
    VectorSchemaErrorCode,
    build_vector_schema_plan,
    render_schema_sql,
    validate_vector_schemas,
)
from .visibility import (
    DEFAULT_VISIBILITY,
    Visibility,
    coerce_bool_flag,
    normalize_visibility,
)
from . import vector_client as vector_client

__all__ = [
    "VectorStore",
    "VectorStoreRouter",
    "TenantScopedVectorStore",
    "get_default_router",
    "vector_client",
    "resolve_ingestion_profile",
    "IngestionContractErrorCode",
    "ensure_embedding_dimensions",
    "map_ingestion_error_to_status",
    "resolve_embedding_profile",
    "ProfileResolverError",
    "ProfileResolverErrorCode",
    "CandidatePoolPolicy",
    "resolve_vector_space_full",
    "resolve_vector_space",
    "VectorSpaceResolverError",
    "VectorSpaceResolverErrorCode",
    "build_vector_schema_plan",
    "render_schema_sql",
    "validate_vector_schemas",
    "VectorSchemaError",
    "VectorSchemaErrorCode",
    "RouterInputError",
    "RouterInputErrorCode",
    "emit_router_validation_failure",
    "map_router_error_to_status",
    "validate_search_inputs",
    "resolve_candidate_pool_policy",
    "Visibility",
    "DEFAULT_VISIBILITY",
    "normalize_visibility",
    "coerce_bool_flag",
    "GuardrailLimits",
    "GuardrailSignals",
    "QuotaLimits",
    "QuotaUsage",
    "FetcherLimits",
]
