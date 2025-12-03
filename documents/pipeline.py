"""Core configuration and state helpers for the document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import import_module
from time import perf_counter
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Protocol,
)
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.logging import get_logger, log_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from documents.contract_utils import (
    normalize_optional_string,
    normalize_string,
    normalize_tenant,
    normalize_workflow_id,
)
from documents.asset_ingestion import AssetIngestionPipeline

from common.assets import AssetIngestPayload

from . import metrics
from .logging_utils import document_log_fields, log_extra_entry, log_extra_exit
from .parsers import (
    DocumentParser,
    ParsedAsset,
    ParsedResult,
    ParsedTextBlock,
    ParserDispatcher,
)
from .processing_graph import (
    DocumentProcessingPhase,
    DocumentProcessingState,
    build_document_processing_graph,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .captioning import MultimodalCaptioner

CollectionKey = Union[str, UUID]


logger = get_logger(__name__)


_CAPTION_SOURCE_CONFIDENCE: Dict[str, float] = {
    "alt_text": 1.0,
    "figure_caption": 0.9,
    "notes": 0.8,
    "context_after": 0.7,
    "context_before": 0.6,
    "origin": 0.5,
}


class ProcessingState(str, Enum):
    """Enumerates the sequential ingestion processing states."""

    INGESTED = "INGESTED"
    PARSED_TEXT = "PARSED_TEXT"
    ASSETS_EXTRACTED = "ASSETS_EXTRACTED"
    CAPTIONED = "CAPTIONED"
    CHUNKED = "CHUNKED"


def _coerce_bool(value: bool, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name}_type")


def _coerce_confidence(value: float, *, field_name: str) -> float:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return numeric
    raise ValueError(f"{field_name}_range")


def _normalise_collection_key(key: CollectionKey) -> str:
    if isinstance(key, UUID):
        return str(key)
    if isinstance(key, str):
        normalized = normalize_string(key)
        if not normalized:
            raise ValueError("caption_min_confidence_collection_empty")
        return normalized.lower()
    raise ValueError("caption_min_confidence_collection_key")


def _normalise_collection_mapping(
    mapping: Optional[Mapping[CollectionKey, float]],
) -> Mapping[str, float]:
    if mapping is None:
        return MappingProxyType({})
    if not isinstance(mapping, Mapping):
        raise ValueError("caption_min_confidence_by_collection_type")
    normalised: dict[str, float] = {}
    for key, value in mapping.items():
        try:
            normalised_key = _normalise_collection_key(key)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(str(exc)) from exc
        normalised[normalised_key] = _coerce_confidence(
            value, field_name="caption_min_confidence"
        )
    return dict(normalised)


@dataclass(frozen=True)
class DocumentPipelineConfig:
    """Configuration contract for ingestion and enrichment pipelines."""

    pdf_safe_mode: bool = True
    caption_min_confidence_default: float = 0.4
    caption_min_confidence_by_collection: Mapping[str, float] = field(
        default_factory=dict
    )
    enable_ocr: bool = False
    enable_notes_in_pptx: bool = True
    emit_empty_slides: bool = True
    enable_asset_captions: bool = True
    ocr_fallback_confidence: float = 0.5
    use_readability_html_extraction: bool = False
    ocr_renderer: Optional[Callable[..., Any]] = field(default=None, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(
            self,
            "pdf_safe_mode",
            _coerce_bool(self.pdf_safe_mode, field_name="pdf_safe_mode"),
        )
        object.__setattr__(
            self,
            "enable_ocr",
            _coerce_bool(self.enable_ocr, field_name="enable_ocr"),
        )
        object.__setattr__(
            self,
            "enable_notes_in_pptx",
            _coerce_bool(self.enable_notes_in_pptx, field_name="enable_notes_in_pptx"),
        )
        object.__setattr__(
            self,
            "emit_empty_slides",
            _coerce_bool(self.emit_empty_slides, field_name="emit_empty_slides"),
        )
        object.__setattr__(
            self,
            "enable_asset_captions",
            _coerce_bool(
                self.enable_asset_captions, field_name="enable_asset_captions"
            ),
        )
        object.__setattr__(
            self,
            "ocr_fallback_confidence",
            _coerce_confidence(
                self.ocr_fallback_confidence,
                field_name="ocr_fallback_confidence",
            ),
        )
        object.__setattr__(
            self,
            "use_readability_html_extraction",
            _coerce_bool(
                self.use_readability_html_extraction,
                field_name="use_readability_html_extraction",
            ),
        )
        object.__setattr__(
            self,
            "caption_min_confidence_default",
            _coerce_confidence(
                self.caption_min_confidence_default,
                field_name="caption_min_confidence_default",
            ),
        )
        object.__setattr__(
            self,
            "caption_min_confidence_by_collection",
            _normalise_collection_mapping(self.caption_min_confidence_by_collection),
        )
        renderer = self.ocr_renderer
        if renderer is not None and not callable(renderer):
            raise ValueError("ocr_renderer_invalid")

    def caption_min_confidence(self, collection_id: Optional[CollectionKey]) -> float:
        """Return the effective caption confidence threshold for a collection.

        ``collection_id`` lookups are case-insensitive for string identifiers and
        UUID inputs should use their canonical string representation.
        """

        if collection_id is not None:
            try:
                key = _normalise_collection_key(collection_id)
            except ValueError:
                return self.caption_min_confidence_default
            candidate = self.caption_min_confidence_by_collection.get(key)
            if candidate is not None:
                return candidate
        return self.caption_min_confidence_default


class DocumentProcessingMetadata(BaseModel):
    """Immutable metadata propagated through ingestion states."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(..., description="Tenant owning the document")
    collection_id: Optional[UUID] = Field(
        default=None, description="Collection identifier the document belongs to"
    )
    document_collection_id: Optional[UUID] = Field(
        default=None,
        description="Logical DocumentCollection identifier associated with the document",
    )
    case_id: Optional[str] = Field(
        default=None, description="Case identifier associated with the document"
    )
    workflow_id: str = Field(
        ..., description="Workflow identifier that processed the document"
    )
    document_id: UUID = Field(..., description="Unique document identifier")
    version: Optional[str] = Field(
        default=None, description="Semantic document version or revision"
    )
    source: Optional[str] = Field(
        default=None, description="Ingestion source for the document"
    )
    created_at: datetime = Field(
        ..., description="UTC timestamp when the document was ingested"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace identifier correlating logs and telemetry",
    )
    span_id: Optional[str] = Field(
        default=None,
        description="Span identifier correlating logs and telemetry",
    )

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalize_tenant(cls, value: str) -> str:
        if value is None:
            raise ValueError("metadata_required_string")
        return normalize_tenant(str(value))

    @field_validator("collection_id", "document_collection_id", mode="before")
    @classmethod
    def _coerce_optional_uuid(cls, value: Optional[UUID] | str) -> Optional[UUID]:
        if value is None:
            return None
        if isinstance(value, UUID):
            return value
        candidate = normalize_optional_string(value)
        if not candidate:
            return None
        return UUID(candidate)

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _normalize_workflow(cls, value: str) -> str:
        if value is None:
            raise ValueError("metadata_required_string")
        return normalize_workflow_id(str(value))

    @field_validator("case_id", "version", "trace_id", "span_id", mode="before")
    @classmethod
    def _normalise_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalise_created_at(cls, value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError("metadata_created_at_type")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("metadata_created_at_naive")
        return value.astimezone(timezone.utc)

    @classmethod
    def from_document(
        cls,
        document,
        *,
        case_id: Optional[str] = None,
        document_collection_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> "DocumentProcessingMetadata":
        """Build metadata from a :class:`NormalizedDocument` instance."""

        try:
            ref = document.ref
            created_at = document.created_at
            source = document.source
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise TypeError("metadata_document_invalid") from exc

        def _coalesce_trace(source_obj: Any) -> None:
            nonlocal trace_id, span_id
            if source_obj is None:
                return
            if isinstance(source_obj, Mapping):
                trace_id = trace_id or source_obj.get("trace_id")
                span_id = span_id or source_obj.get("span_id")
                return
            candidate_trace = getattr(source_obj, "trace_id", None)
            candidate_span = getattr(source_obj, "span_id", None)
            trace_id = trace_id or candidate_trace
            span_id = span_id or candidate_span

        meta = getattr(document, "meta", None)
        _coalesce_trace(document)
        _coalesce_trace(meta)
        graph_state = getattr(meta, "graph_state", None) if meta is not None else None
        _coalesce_trace(graph_state)

        workflow_id = getattr(ref, "workflow_id", None)
        if workflow_id is None and hasattr(document, "meta"):
            workflow_id = getattr(document.meta, "workflow_id", None)
        if workflow_id is None:
            raise ValueError("metadata_workflow_missing")

        if document_collection_id is None:
            document_collection_id = getattr(ref, "document_collection_id", None)
        if document_collection_id is None:
            meta = getattr(document, "meta", None)
            document_collection_id = getattr(meta, "document_collection_id", None)

        return cls(
            tenant_id=ref.tenant_id,
            collection_id=getattr(ref, "collection_id", None),
            document_collection_id=document_collection_id,
            case_id=case_id,
            workflow_id=workflow_id,
            document_id=ref.document_id,
            version=getattr(ref, "version", None),
            source=source,
            created_at=created_at,
            trace_id=trace_id,
            span_id=span_id,
        )


@dataclass(frozen=True)
class DocumentProcessingContext:
    """Tracks processing state transitions while preserving metadata."""

    metadata: DocumentProcessingMetadata
    state: ProcessingState = ProcessingState.INGESTED
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if self.metadata.trace_id is not None:
            object.__setattr__(self, "trace_id", self.metadata.trace_id)
        if self.metadata.span_id is not None:
            object.__setattr__(self, "span_id", self.metadata.span_id)

    def __repr__(self) -> str:  # pragma: no cover - human convenience
        return (
            "DocumentProcessingContext("
            f"state={self.state.value!r}, "
            f"metadata=DocumentProcessingMetadata(tenant_id={self.metadata.tenant_id!r}, "
            f"collection_id={self.metadata.collection_id!r}, "
            f"document_collection_id={self.metadata.document_collection_id!r}, "
            f"case_id={self.metadata.case_id!r}, "
            f"workflow_id={self.metadata.workflow_id!r}, "
            f"document_id={self.metadata.document_id!r}, "
            f"version={self.metadata.version!r}, "
            f"source={self.metadata.source!r}, "
            f"created_at={self.metadata.created_at!r}, "
            f"trace_id={self.metadata.trace_id!r}, "
            f"span_id={self.metadata.span_id!r})), "
            f"trace_id={self.trace_id!r}, "
            f"span_id={self.span_id!r})"
        )

    def transition(
        self, new_state: ProcessingState | str
    ) -> "DocumentProcessingContext":
        """Return a new context with ``new_state`` while preserving metadata."""

        try:
            state = (
                new_state
                if isinstance(new_state, ProcessingState)
                else ProcessingState(new_state)
            )
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("processing_state_invalid") from exc
        return DocumentProcessingContext(
            metadata=self.metadata,
            state=state,
            trace_id=self.trace_id,
            span_id=self.span_id,
        )

    @classmethod
    def from_document(
        cls,
        document,
        *,
        case_id: Optional[str] = None,
        document_collection_id: Optional[UUID] = None,
        initial_state: ProcessingState | str = ProcessingState.INGESTED,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> "DocumentProcessingContext":
        metadata = DocumentProcessingMetadata.from_document(
            document,
            case_id=case_id,
            document_collection_id=document_collection_id,
            trace_id=trace_id,
            span_id=span_id,
        )
        if not isinstance(initial_state, ProcessingState):
            initial_state = ProcessingState(initial_state)
        return cls(
            metadata=metadata,
            state=initial_state,
            trace_id=metadata.trace_id,
            span_id=metadata.span_id,
        )


@dataclass(frozen=True)
class DocumentContracts:
    """Resolved contract types required by ingestion pipelines."""

    normalized_document: type
    document_ref: type
    asset: type
    asset_ref: type


@dataclass(frozen=True)
class DocumentComponents:
    """Resolved infrastructure components used by ingestion pipelines."""

    repository: type
    storage: type
    captioner: type


class DocumentChunker(Protocol):
    """Protocol describing chunking implementations used by the orchestrator."""

    def chunk(
        self,
        document: Any,
        parsed: ParsedResult,
        *,
        context: DocumentProcessingContext,
        config: DocumentPipelineConfig,
    ) -> Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
        """Return chunk payloads and statistics for ``document``."""


@dataclass(frozen=True)
class DocumentChunkArtifact:
    """Immutable container encapsulating chunking results."""

    context: DocumentProcessingContext
    chunks: Tuple[Mapping[str, Any], ...]
    statistics: Mapping[str, Any]

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        serialised: list[Mapping[str, Any]] = []
        for chunk in self.chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("chunk_artifact_chunks")
            serialised.append(dict(chunk))
        object.__setattr__(self, "chunks", tuple(serialised))

        if not isinstance(self.statistics, Mapping):
            raise ValueError("chunk_artifact_statistics")
        object.__setattr__(self, "statistics", dict(self.statistics))


@dataclass(frozen=True)
class DocumentProcessingOutcome:
    """Return type of the orchestration pipeline."""

    document: Any
    context: DocumentProcessingContext
    parse_artifact: Optional[DocumentParseArtifact] = None
    chunk_artifact: Optional[DocumentChunkArtifact] = None


_STATE_SEQUENCE: Tuple[ProcessingState, ...] = (
    ProcessingState.INGESTED,
    ProcessingState.PARSED_TEXT,
    ProcessingState.ASSETS_EXTRACTED,
    ProcessingState.CAPTIONED,
    ProcessingState.CHUNKED,
)


def _state_rank(state: Optional[ProcessingState | str]) -> int:
    if state is None:
        return 0
    if not isinstance(state, ProcessingState):
        try:
            state = ProcessingState(str(state))
        except ValueError:
            return 0
    return _STATE_SEQUENCE.index(state)


def _max_state(
    first: ProcessingState, second: Optional[ProcessingState | str]
) -> ProcessingState:
    if second is None:
        return first
    if not isinstance(second, ProcessingState):
        try:
            second = ProcessingState(str(second))
        except ValueError:
            return first
    return first if _state_rank(first) >= _state_rank(second) else second


def _state_from_stats(stats: Optional[Mapping[str, Any]]) -> Optional[ProcessingState]:
    if not stats:
        return None
    highest: Optional[ProcessingState] = None
    for key in ("chunk.state", "caption.state", "assets.state", "parse.state"):
        value = stats.get(key)
        if value is None:
            continue
        try:
            candidate = ProcessingState(str(value))
        except ValueError:
            continue
        if highest is None or _state_rank(candidate) > _state_rank(highest):
            highest = candidate
    return highest


def _update_document_stats(
    document: Any,
    updates: Mapping[str, Any],
    *,
    repository: Any,
    workflow_id: str,
) -> Any:
    meta = document.meta
    current = dict(getattr(meta, "parse_stats", {}) or {})
    current.update(dict(updates))
    meta_copy = meta.model_copy(update={"parse_stats": current}, deep=True)
    document_copy = document.model_copy(update={"meta": meta_copy}, deep=True)
    return repository.upsert(document_copy, workflow_id=workflow_id)


def _normalise_workflow_label(value: Optional[str]) -> str:
    candidate = (value or "").strip()
    return candidate or "unknown"


def _observe_counts(
    *,
    workflow_id: str,
    blocks: int,
    assets: int,
    ocr_triggers: int,
) -> None:
    workflow_label = _normalise_workflow_label(workflow_id)
    metrics.PIPELINE_BLOCKS_TOTAL.labels(workflow_id=workflow_label).inc(blocks)
    metrics.PIPELINE_ASSETS_TOTAL.labels(workflow_id=workflow_label).inc(assets)
    if ocr_triggers:
        metrics.PIPELINE_OCR_TRIGGER_TOTAL.labels(workflow_id=workflow_label).inc(
            ocr_triggers
        )


def _observe_caption_metrics(
    *,
    workflow_id: str,
    hits: int,
    attempts: int,
) -> None:
    workflow_label = _normalise_workflow_label(workflow_id)
    if attempts:
        ratio = hits / attempts
        metrics.PIPELINE_CAPTION_HIT_RATIO.labels(workflow_id=workflow_label).observe(
            ratio
        )
    metrics.PIPELINE_CAPTION_ATTEMPTS_TOTAL.labels(workflow_id=workflow_label).inc(
        attempts
    )
    if hits:
        metrics.PIPELINE_CAPTION_HITS_TOTAL.labels(workflow_id=workflow_label).inc(hits)


def _run_phase(
    span_name: str,
    metric_event: str,
    *,
    workflow_id: str,
    attributes: Optional[Mapping[str, Any]] = None,
    action: Callable[[], Any],
) -> Any:
    tracer = trace.get_tracer(__name__)
    start = perf_counter()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                span.set_attribute(key, value)
        try:
            result = action()
        except Exception as exc:  # pragma: no cover - error path exercised in tests
            duration = (perf_counter() - start) * 1000.0
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            metrics.observe_event(
                metric_event, "error", duration, workflow_id=workflow_id
            )
            raise
        else:
            duration = (perf_counter() - start) * 1000.0
            span.set_status(Status(StatusCode.OK))
            metrics.observe_event(metric_event, "ok", duration, workflow_id=workflow_id)
            return result


@dataclass(frozen=True)
class DocumentParseArtifact:
    """Immutable container returned after persisting parser results."""

    parsed_context: DocumentProcessingContext
    asset_context: DocumentProcessingContext
    text_blocks: Tuple[Mapping[str, Any], ...]
    asset_refs: Tuple[object, ...]
    statistics: Mapping[str, Any]

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        serialised_blocks: list[Mapping[str, Any]] = []
        for block in self.text_blocks:
            if not isinstance(block, Mapping):
                raise ValueError("parse_artifact_text_blocks")
            serialised_blocks.append(dict(block))
        object.__setattr__(self, "text_blocks", tuple(serialised_blocks))

        object.__setattr__(self, "asset_refs", tuple(self.asset_refs))

        if not isinstance(self.statistics, Mapping):
            raise ValueError("parse_artifact_statistics")
        object.__setattr__(self, "statistics", dict(self.statistics))


def _import_guarded(module_names: Union[str, Sequence[str]], *, code: str):
    candidates = (
        (module_names,) if isinstance(module_names, str) else tuple(module_names)
    )
    last_exc: Optional[ModuleNotFoundError] = None
    for module_name in candidates:
        try:
            return import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
            last_exc = exc
            continue
    raise RuntimeError(code) from last_exc


def _require_attribute(module, attribute: str, *, code: str) -> object:
    try:
        value = getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - propagate stable code
        raise RuntimeError(code) from exc
    if value is None:
        raise RuntimeError(code)
    return value


def require_document_contracts() -> DocumentContracts:
    """Return core contract classes, raising on missing dependencies."""

    contracts = _import_guarded(
        "documents.contracts", code="contract_missing_documents_contracts_module"
    )
    normalized_document = _require_attribute(
        contracts, "NormalizedDocument", code="contract_missing_normalized_document"
    )
    document_ref = _require_attribute(
        contracts, "DocumentRef", code="contract_missing_document_ref"
    )
    asset = _require_attribute(contracts, "Asset", code="contract_missing_asset")
    asset_ref = _require_attribute(
        contracts, "AssetRef", code="contract_missing_asset_ref"
    )
    return DocumentContracts(
        normalized_document=normalized_document,
        document_ref=document_ref,
        asset=asset,
        asset_ref=asset_ref,
    )


def require_document_components() -> DocumentComponents:
    """Return repository, storage and captioner types, validating their presence."""

    repository_module = _import_guarded(
        ("documents.repository", "documents.repositories"),
        code="contract_missing_documents_repository_module",
    )
    storage_module = _import_guarded(
        ("documents.storage", "documents.storages"),
        code="contract_missing_documents_storage_module",
    )
    captioning_module = _import_guarded(
        ("documents.captioning", "documents.captions"),
        code="contract_missing_documents_captioning_module",
    )

    repository = _require_attribute(
        repository_module,
        "DocumentsRepository",
        code="contract_missing_documents_repository",
    )
    storage = _require_attribute(
        storage_module, "Storage", code="contract_missing_storage"
    )
    captioner = _require_attribute(
        captioning_module,
        "MultimodalCaptioner",
        code="contract_missing_captioner",
    )
    return DocumentComponents(
        repository=repository, storage=storage, captioner=captioner
    )


def _serialise_text_block(block: ParsedTextBlock) -> Mapping[str, Any]:
    payload: Dict[str, Any] = {
        "text": block.text,
        "kind": block.kind,
        "section_path": list(block.section_path) if block.section_path else None,
        "page_index": block.page_index,
        "table_meta": dict(block.table_meta) if block.table_meta is not None else None,
        "language": block.language,
    }
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        parent_ref = metadata.get("parent_ref")
        if parent_ref:
            payload["parent_ref"] = parent_ref
        locator = metadata.get("locator")
        if locator:
            payload["locator"] = locator
    return payload


def _external_kind_for_uri(uri: str) -> str:
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https", "s3", "gcs"}:
        return scheme
    return "http"


def _ensure_context_matches_document(
    context: DocumentProcessingContext, document: Any
) -> None:
    ref = getattr(document, "ref", None)
    if ref is None:
        raise ValueError("document_missing_ref")
    metadata = context.metadata
    if metadata.tenant_id != getattr(ref, "tenant_id", None):
        raise ValueError("context_document_mismatch")
    if metadata.workflow_id != getattr(ref, "workflow_id", None):
        raise ValueError("context_document_mismatch")
    if metadata.document_id != getattr(ref, "document_id", None):
        raise ValueError("context_document_mismatch")
    if metadata.collection_id != getattr(ref, "collection_id", None):
        raise ValueError("context_document_mismatch")
    ref_collection = getattr(ref, "document_collection_id", None)
    if metadata.document_collection_id != ref_collection:
        raise ValueError("context_document_mismatch")
    if metadata.version != getattr(ref, "version", None):
        raise ValueError("context_document_mismatch")
    if metadata.source != getattr(document, "source", None):
        raise ValueError("context_document_mismatch")
    if metadata.created_at != getattr(document, "created_at", None):
        raise ValueError("context_document_mismatch")


def _persist_asset(
    *,
    index: int,
    parsed_asset: ParsedAsset,
    metadata: DocumentProcessingMetadata,
    repository: Any,
    storage: Any,
    contracts: DocumentContracts,
) -> object:
    """Persist asset using AssetIngestionPipeline.

    Refactored from 188-line god-function to orchestration delegate.
    Business logic now in:
    - AssetIngestionPipeline: Main orchestration
    - CaptionResolver: Caption prioritization
    - BlobStorageAdapter: Storage & hashing
    """
    pipeline = AssetIngestionPipeline(
        repository=repository,
        storage=storage,
        contracts=contracts,
    )

    ingest_payload = AssetIngestPayload(
        media_type=parsed_asset.media_type,
        metadata=getattr(parsed_asset, "metadata", {}) or {},
        content=parsed_asset.content,
        file_uri=parsed_asset.file_uri,
        page_index=parsed_asset.page_index,
        bbox=parsed_asset.bbox,
        context_before=parsed_asset.context_before,
        context_after=parsed_asset.context_after,
    )

    return pipeline.persist_asset(
        index=index,
        asset_payload=ingest_payload,
        tenant_id=metadata.tenant_id,
        workflow_id=metadata.workflow_id,
        document_id=metadata.document_id,
        collection_id=metadata.collection_id,
        created_at=metadata.created_at,
    )


def persist_parsed_document(
    context: DocumentProcessingContext,
    document: Any,
    parsed: ParsedResult,
    *,
    repository: Any,
    storage: Any,
    contracts: Optional[DocumentContracts] = None,
) -> DocumentParseArtifact:
    """Persist parser output and return a chunker-friendly artefact."""

    _ensure_context_matches_document(context, document)

    contracts = contracts or require_document_contracts()

    stats = dict(parsed.statistics)
    stats["parse.state"] = ProcessingState.PARSED_TEXT.value
    stats["assets.state"] = ProcessingState.ASSETS_EXTRACTED.value
    text_blocks = tuple(_serialise_text_block(block) for block in parsed.text_blocks)

    document_copy = document.model_copy(deep=True)
    meta_copy = document_copy.meta.model_copy(update={"parse_stats": stats}, deep=True)
    document_copy.meta = meta_copy
    repository.upsert(document_copy, workflow_id=context.metadata.workflow_id)

    asset_refs: list[object] = []
    for index, asset in enumerate(parsed.assets):
        stored = _persist_asset(
            index=index,
            parsed_asset=asset,
            metadata=context.metadata,
            repository=repository,
            storage=storage,
            contracts=contracts,
        )
        asset_refs.append(stored.ref)

    parsed_context = context.transition(ProcessingState.PARSED_TEXT)
    asset_context = parsed_context.transition(ProcessingState.ASSETS_EXTRACTED)

    return DocumentParseArtifact(
        parsed_context=parsed_context,
        asset_context=asset_context,
        text_blocks=text_blocks,
        asset_refs=tuple(asset_refs),
        statistics=stats,
    )


class DocumentProcessingOrchestrator:
    """Stateful orchestrator executing the ingestion pipeline phases."""

    def __init__(
        self,
        *,
        parser: DocumentParser | ParserDispatcher | Any,
        repository: Any,
        storage: Any,
        captioner: "MultimodalCaptioner",
        chunker: DocumentChunker,
        config: Optional[DocumentPipelineConfig] = None,
    ) -> None:
        parse_fn = getattr(parser, "parse", None)
        if not callable(parse_fn):
            raise TypeError("parser_missing_parse")
        chunk_fn = getattr(chunker, "chunk", None)
        if not callable(chunk_fn):
            raise TypeError("chunker_missing_chunk")
        self._parser = parser
        self.repository = repository
        self.storage = storage
        self.captioner = captioner
        self.chunker = chunker
        self.config = config or DocumentPipelineConfig()
        self._graph = build_document_processing_graph(
            parser=self._parser,
            repository=self.repository,
            storage=self.storage,
            captioner=self.captioner,
            chunker=self.chunker,
        )

    def process(
        self,
        document: Any,
        *,
        context: Optional[DocumentProcessingContext] = None,
        case_id: Optional[str] = None,
        run_until: DocumentProcessingPhase | str | None = None,
    ) -> DocumentProcessingOutcome:
        document_stats = dict(getattr(document.meta, "parse_stats", {}) or {})
        stats_state = _state_from_stats(document_stats)

        if context is None:
            initial_state = stats_state or ProcessingState.INGESTED
            context = DocumentProcessingContext.from_document(
                document,
                case_id=case_id,
                initial_state=initial_state,
            )
        else:
            _ensure_context_matches_document(context, document)
            if stats_state and _state_rank(stats_state) > _state_rank(context.state):
                context = context.transition(stats_state)

        metadata = context.metadata
        workflow_id = metadata.workflow_id

        with log_context(
            trace_id=context.trace_id,
            span_id=context.span_id,
            tenant=metadata.tenant_id,
            collection_id=(
                str(metadata.collection_id) if metadata.collection_id else None
            ),
            workflow_id=workflow_id,
        ):
            log_extra_entry(**document_log_fields(document), state=context.state.value)
            graph_state = DocumentProcessingState(
                document=document,
                config=self.config,
                context=context,
                run_until=run_until,
            )
            final_state: DocumentProcessingState = graph_state
            try:
                final_state = self._graph.invoke(graph_state)
            finally:
                final_context = final_state.context.state
                log_extra_exit(final_state=final_context.value)

        return DocumentProcessingOutcome(
            document=final_state.document,
            context=final_state.context,
            parse_artifact=final_state.parse_artifact,
            chunk_artifact=final_state.chunk_artifact,
        )


__all__ = [
    "DocumentChunkArtifact",
    "DocumentChunker",
    "DocumentComponents",
    "DocumentContracts",
    "DocumentParseArtifact",
    "DocumentPipelineConfig",
    "DocumentProcessingOrchestrator",
    "DocumentProcessingContext",
    "DocumentProcessingMetadata",
    "DocumentProcessingOutcome",
    "ProcessingState",
    "require_document_components",
    "require_document_contracts",
    "persist_parsed_document",
]
