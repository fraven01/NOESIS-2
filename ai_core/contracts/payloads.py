"""Canonical payload models for crawler ingestion graph."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_serializer


# ============================================================================
# Guardrails
# ============================================================================


class GuardrailLimitsData(BaseModel):
    """Serializable guardrail limits."""

    max_document_bytes: int | None = None
    processing_time_limit_ms: float | None = None
    mime_blacklist: tuple[str, ...] = ()
    host_blocklist: tuple[str, ...] = ()
    tenant_quota_max_docs: int | None = None
    tenant_quota_max_bytes: int | None = None
    host_quota_max_docs: int | None = None
    host_quota_max_bytes: int | None = None

    model_config = ConfigDict(frozen=True)


class GuardrailSignalsData(BaseModel):
    """Serializable guardrail signals."""

    tenant_id: str
    provider: str | None = None
    canonical_source: str | None = None
    host: str | None = None
    document_bytes: int = 0
    mime_type: str | None = None
    processing_time_ms: float | None = None
    tenant_usage_docs: int = 0
    tenant_usage_bytes: int = 0
    host_usage_docs: int = 0
    host_usage_bytes: int = 0

    model_config = ConfigDict(frozen=True)


class GuardrailPayload(BaseModel):
    """Complete guardrail state (replaces GuardrailSerde)."""

    decision: str
    reason: str
    allowed: bool
    policy_events: tuple[str, ...] = ()
    limits: GuardrailLimitsData | None = None
    signals: GuardrailSignalsData | None = None
    attributes: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


# ============================================================================
# Fetch
# ============================================================================


class FetchRequestData(BaseModel):
    """Fetch request metadata."""

    canonical_source: str
    metadata: dict[str, str] = Field(default_factory=dict)
    politeness_host: str | None = None
    politeness_user_agent: str | None = None
    politeness_crawl_delay: float | None = None

    model_config = ConfigDict(frozen=True)


class FetchPayload(BaseModel):
    """Unified fetch payload (manual + real have same structure)."""

    request: FetchRequestData
    status_code: int
    body: bytes
    headers: dict[str, str] = Field(default_factory=dict)
    elapsed_ms: float
    retries: int = 0
    retry_reason: str | None = None
    downloaded_bytes: int
    backoff_total_ms: float = 0.0
    max_bytes_limit: int | None = None
    timeout_seconds: float | None = None
    failure_reason: str | None = None
    failure_temporary: bool | None = None

    model_config = ConfigDict(frozen=True)

    @field_serializer("body", when_used="json")
    def serialize_body_as_base64(self, value: bytes) -> str:
        """Serialize binary body as base64 for JSON compatibility."""
        import base64

        return base64.b64encode(value).decode("ascii")


# ============================================================================
# Frontier
# ============================================================================


class FrontierData(BaseModel):
    """Frontier decision input."""

    host: str
    path: str
    provider: str
    breadcrumbs: tuple[str, ...] = ()
    policy_events: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True)


# ============================================================================
# Delta
# ============================================================================


class DeltaPayload(BaseModel):
    """Delta decision payload."""

    decision: str
    reason: str
    content_hash: str | None = None
    version: int | None = None
    changed_fields: tuple[str, ...] = ()
    policy_events: tuple[str, ...] = ()
    attributes: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


# ============================================================================
# Embedding
# ============================================================================


class EmbeddingPayload(BaseModel):
    """Embedding result payload."""

    status: str
    chunks_inserted: int
    embedding_profile: str
    vector_space_id: str
    tenant_id: str
    case_id: str
    document_id: str
    workflow_id: str | None = None
    collection_id: str | None = None

    model_config = ConfigDict(frozen=True)


# ============================================================================
# Graph State
# ============================================================================


class ControlFlags(BaseModel):
    """Control flags for graph execution."""

    dry_run: bool = False
    shadow_mode: bool = False
    snapshot_enabled: bool = False
    snapshot_label: str | None = None
    review_required: bool = False
    force_retire: bool = False
    recompute_delta: bool = False

    model_config = ConfigDict(frozen=True)


class CrawlerGraphState(BaseModel):
    """Typed crawler ingestion graph state (replaces dict)."""

    tenant_id: str
    case_id: str
    workflow_id: str
    document_id: str
    collection_id: str | None = None
    normalized_document_input: object
    frontier: FrontierData
    fetch: FetchPayload
    guardrails: GuardrailPayload
    control: ControlFlags
    baseline_checksum: str | None = None
    baseline_version: int | None = None
    baseline_lifecycle_state: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    external_id: str | None = None
    origin_uri: str | None = None
    provider: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Completion
# ============================================================================


class CompletionPayload(BaseModel):
    """Graph completion payload (for finish node + external APIs)."""

    normalized_document: dict[str, object]
    delta: DeltaPayload
    guardrails: GuardrailPayload
    embedding: EmbeddingPayload | None = None
    pipeline_phase: str | None = None
    pipeline_error: str | None = None
    pipeline_run_until: str | None = None
    failure: dict[str, object] | None = None

    model_config = ConfigDict(frozen=True)
