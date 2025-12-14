"""LangGraph inspired orchestration for crawler ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timedelta
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import ValidationError, BaseModel
from uuid import UUID, uuid4

from ai_core.api import EmbeddingResult
from ai_core import api as ai_core_api
from ai_core.contracts.scope import ScopeContext
from ai_core.contracts.payloads import (
    CompletionPayload,
    FrontierData,
    GuardrailLimitsData,
    GuardrailPayload as GuardrailStatePayload,
    GuardrailSignalsData,
)
from ai_core.graphs.transition_contracts import (
    DeltaSection,
    EmbeddingSection,
    GraphTransition,
    GuardrailSection,
    LifecycleSection,
    PipelineSection,
    StandardTransitionResult,
    build_delta_section,
    build_embedding_section,
    build_guardrail_section,
    build_lifecycle_section,
)
from ai_core.infra import object_store
from ai_core.infra import observability as observability_module
from ai_core.infra.observability import (
    emit_event,
    observe_span,
    update_observation,
)
from ai_core.rag.guardrails import (
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)
from ai_core.rag.ingestion_contracts import (
    ChunkMeta,
    IngestionProfileResolution,
    resolve_ingestion_profile,
)
from documents.service_facade import ingest_document
from documents import metrics as document_metrics
from documents.api import LifecycleStatusUpdate, NormalizedDocumentPayload
from documents.contracts import NormalizedDocument
from documents.normalization import document_payload_bytes, normalized_primary_text
from documents.pipeline import (
    DocumentPipelineConfig,
    DocumentProcessingContext,
    require_document_components,
)
from documents.processing_graph import (
    DocumentProcessingPhase,
    DocumentProcessingState,
    build_document_processing_graph,
)
from documents.repository import DocumentsRepository
from documents.parsers import ParsedResult, ParserDispatcher, ParserRegistry
from documents.cli import SimpleDocumentChunker
from documents import (
    DocxDocumentParser,
    HtmlDocumentParser,
    MarkdownDocumentParser,
    PdfDocumentParser,
    PptxDocumentParser,
    TextDocumentParser,
)
from .document_service import (
    DocumentLifecycleService,
    DocumentPersistenceService,
    DocumentsApiLifecycleService,
    DocumentsRepositoryAdapter,
)

StateMapping = Mapping[str, Any] | MutableMapping[str, Any]


@dataclass(frozen=True)
class GraphNode:
    """Tie a node name to an execution callable."""

    name: str
    runner: Callable[[Dict[str, Any]], Tuple[GraphTransition, bool]]

    def execute(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        return self.runner(state)


@dataclass(frozen=True)
class ChunkComputation:
    """Captured context required to embed chunked crawler documents."""

    meta: Dict[str, Any]
    chunks_path: str
    text_path: str
    profile: IngestionProfileResolution
    blocks_path: Optional[str] = None
    chunk_count: Optional[int] = None


def _transition(
    *,
    phase: str,
    decision: str,
    reason: str,
    severity: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    pipeline: Optional[PipelineSection] = None,
    lifecycle: Optional[LifecycleSection] = None,
    delta: Optional[DeltaSection | ai_core_api.DeltaDecision] = None,
    guardrail: Optional[GuardrailSection | ai_core_api.GuardrailDecision] = None,
    embedding: Optional[EmbeddingSection | EmbeddingResult] = None,
) -> GraphTransition:
    delta_section: Optional[DeltaSection]
    if isinstance(delta, DeltaSection) or delta is None:
        delta_section = delta
    else:
        delta_section = build_delta_section(delta)

    guardrail_section: Optional[GuardrailSection]
    if isinstance(guardrail, GuardrailSection) or guardrail is None:
        guardrail_section = guardrail
    else:
        guardrail_section = build_guardrail_section(guardrail)

    embedding_section: Optional[EmbeddingSection]
    if isinstance(embedding, EmbeddingSection) or embedding is None:
        embedding_section = embedding
    else:
        embedding_section = build_embedding_section(embedding)

    result = StandardTransitionResult(
        phase=phase,  # type: ignore[arg-type]
        decision=decision,
        reason=reason,
        severity=severity or "info",
        pipeline=pipeline,
        lifecycle=lifecycle,
        delta=delta_section,
        guardrail=guardrail_section,
        embedding=embedding_section,
        context=dict(context or {}),
    )
    return GraphTransition(result)


class CrawlerIngestionGraph:
    """Minimal orchestration graph coordinating crawler ingestion."""

    def __init__(
        self,
        *,
        document_service: DocumentLifecycleService = DocumentsApiLifecycleService(),
        repository: DocumentsRepository | None = None,
        document_persistence: DocumentPersistenceService | None = None,
        guardrail_enforcer: Callable[
            ..., ai_core_api.GuardrailDecision
        ] = ai_core_api.enforce_guardrails,
        delta_decider: Callable[
            ..., ai_core_api.DeltaDecision
        ] = ai_core_api.decide_delta,
        embedding_handler: Callable[
            ..., EmbeddingResult
        ] = ai_core_api.trigger_embedding,
        completion_builder: Callable[
            ..., Mapping[str, Any]
        ] = ai_core_api.build_completion_payload,
        event_emitter: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        parser_dispatcher: ParserDispatcher | None = None,
        storage: Any | None = None,
        captioner: Any | None = None,
        chunker: Any | None = None,
        pipeline_config: DocumentPipelineConfig | None = None,
    ) -> None:
        self._document_service = document_service
        persistence_candidate = document_persistence
        if persistence_candidate is None:
            service_repository = getattr(document_service, "repository", None)
            if (
                hasattr(document_service, "upsert_normalized")
                and service_repository is not None
            ):
                persistence_candidate = document_service  # type: ignore[assignment]
            else:
                persistence_candidate = DocumentsRepositoryAdapter(
                    repository=repository
                )
        if repository is None and hasattr(persistence_candidate, "repository"):
            repository = getattr(persistence_candidate, "repository")
        self._repository = repository
        self._document_persistence = persistence_candidate
        self._guardrail_enforcer = guardrail_enforcer
        self._delta_decider = delta_decider
        self._embedding_handler = embedding_handler
        self._completion_builder = completion_builder
        self._event_emitter = event_emitter

        components = require_document_components()

        if parser_dispatcher is None:
            registry = ParserRegistry(
                [
                    MarkdownDocumentParser(),
                    HtmlDocumentParser(),
                    DocxDocumentParser(),
                    PptxDocumentParser(),
                    PdfDocumentParser(),
                    TextDocumentParser(),
                ]
            )
            parser_dispatcher = ParserDispatcher(registry)
        self._parser_dispatcher = parser_dispatcher

        if storage is None and self._repository is not None:
            storage = getattr(self._repository, "storage", None)
            if storage is None:
                storage = getattr(self._repository, "_storage", None)
        if storage is None:
            # Use concrete ObjectStoreStorage instead of abstract Storage class
            from documents.storage import ObjectStoreStorage

            storage = ObjectStoreStorage()
        self._storage = storage

        if captioner is None:
            captioner_cls = components.captioner
            try:
                captioner = captioner_cls()  # type: ignore[call-arg]
            except Exception:
                captioner = captioner_cls
        self._captioner = captioner

        if chunker is None:
            chunker = SimpleDocumentChunker()
        self._chunker = chunker

        self._pipeline_config = pipeline_config or DocumentPipelineConfig()

        if self._repository is None:
            raise RuntimeError("documents_repository_not_configured")
        if self._storage is None:
            raise RuntimeError("documents_storage_not_configured")
        if self._captioner is None:
            raise RuntimeError("documents_captioner_not_configured")

        self._document_graph = build_document_processing_graph(
            parser=self._parser_dispatcher,
            repository=self._repository,
            storage=self._storage,
            captioner=self._captioner,
            chunker=self._chunker,
        )

        self.upsert_handler: Optional[Callable[[Any], Any]] = None
        self._dedupe_index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}

    def _normalized_from_state(
        self, state: Mapping[str, Any]
    ) -> Optional[NormalizedDocumentPayload]:
        artifacts = state.get("artifacts")
        if isinstance(artifacts, Mapping):
            candidate = artifacts.get("normalized_document")
            if isinstance(candidate, NormalizedDocumentPayload):
                return candidate
        return None

    def _collect_span_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        containers: list[Mapping[str, Any]] = []

        meta_payload = state.get("meta")
        if isinstance(meta_payload, Mapping):
            containers.append(meta_payload)
        containers.append(state)

        raw_document = state.get("raw_document")
        if isinstance(raw_document, Mapping):
            containers.append(raw_document)
            raw_meta = raw_document.get("metadata")
            if isinstance(raw_meta, Mapping):
                containers.append(raw_meta)

        def _first(key: str) -> Optional[Any]:
            for container in containers:
                value = container.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        continue
                    return stripped
                return value
            return None

        for key in ("tenant_id", "case_id", "trace_id", "workflow_id"):
            candidate = _first(key)
            if candidate is not None:
                metadata.setdefault(key, candidate)

        if "document_id" not in metadata:
            for container in containers:
                for field in ("document_id", "external_id", "id"):
                    value = container.get(field)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        stripped = value.strip()
                        if not stripped:
                            continue
                        value = stripped
                    metadata.setdefault("document_id", value)
                    if "document_id" in metadata:
                        break
                if "document_id" in metadata:
                    break

        normalized = self._normalized_from_state(state)
        if normalized is not None:
            metadata.setdefault("tenant_id", normalized.tenant_id)
            metadata.setdefault("document_id", normalized.document_id)
            workflow = getattr(normalized.document.ref, "workflow_id", None)
            if workflow:
                metadata.setdefault("workflow_id", workflow)
            normalized_meta = normalized.metadata
            if isinstance(normalized_meta, Mapping):
                case_candidate = normalized_meta.get("case_id")
                if isinstance(case_candidate, str):
                    case_candidate = case_candidate.strip()
                if case_candidate:
                    metadata.setdefault("case_id", case_candidate)

        graph_run_id = state.get("graph_run_id")
        if isinstance(graph_run_id, str) and graph_run_id.strip():
            metadata.setdefault("graph_run_id", graph_run_id.strip())

        return {
            key: value for key, value in metadata.items() if value not in (None, "")
        }

    def _annotate_span(
        self,
        state: Dict[str, Any],
        *,
        phase: str,
        transition: Optional[GraphTransition] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        metadata = self._collect_span_metadata(state)
        metadata["phase"] = phase
        if transition is not None:
            metadata.setdefault("decision", transition.decision)
            metadata.setdefault("reason", transition.reason)
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                metadata[key] = value
        if metadata:
            span = observability_module._get_current_span()
            span_name = getattr(span, "name", None) if span is not None else None
            update_observation(metadata=metadata)
            expected_name = f"crawler.ingestion.{phase}"
            if span_name == expected_name:
                recorded = state.setdefault("_span_phases", set())
                recorded.add(phase)

    def _with_transition_metadata(
        self, transition: GraphTransition, state: Dict[str, Any]
    ) -> GraphTransition:
        metadata = self._transition_metadata(state)
        if not metadata:
            return transition
        return transition.with_context(metadata)

    @staticmethod
    def _ingestion_tasks():
        from ai_core import tasks as ingestion_tasks

        return ingestion_tasks

    def _resolve_chunk_case_id(
        self, state: Mapping[str, Any], normalized: NormalizedDocumentPayload
    ) -> str:
        candidates = [state.get("case_id")]
        meta_payload = state.get("meta")
        if isinstance(meta_payload, Mapping):
            candidates.append(meta_payload.get("case_id"))
        normalized_meta = normalized.metadata
        if isinstance(normalized_meta, Mapping):
            candidates.append(normalized_meta.get("case_id"))
        external_ref = getattr(normalized.document.meta, "external_ref", None) or {}
        candidates.append(external_ref.get("case_id"))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                text = str(candidate).strip()
            except Exception:
                continue
            if text:
                return text
        return "default"

    def _resolve_chunk_external_id(self, normalized: NormalizedDocumentPayload) -> str:
        external_ref = getattr(normalized.document.meta, "external_ref", None) or {}
        candidate = external_ref.get("external_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        document_id = getattr(normalized.document.ref, "document_id", None)
        if document_id is not None:
            return str(document_id)
        return normalized.checksum

    def _resolve_embedding_profile_binding(
        self, embedding_state: Mapping[str, Any]
    ) -> IngestionProfileResolution:
        profile_candidate = embedding_state.get("profile")
        if isinstance(profile_candidate, str):
            profile_key = profile_candidate.strip() or "standard"
        elif profile_candidate is None:
            profile_key = "standard"
        else:
            profile_key = str(profile_candidate).strip() or "standard"
        return resolve_ingestion_profile(profile_key)

    def _prepare_chunk_meta(
        self,
        state: Dict[str, Any],
        normalized: NormalizedDocumentPayload,
        embedding_state: Mapping[str, Any],
        profile_binding: IngestionProfileResolution,
    ) -> Dict[str, Any]:
        tenant_id = str(normalized.tenant_id)
        case_id = self._resolve_chunk_case_id(state, normalized)
        external_id = self._resolve_chunk_external_id(normalized)
        workflow_id = getattr(normalized.document.ref, "workflow_id", None)
        document_id = normalized.document_id

        meta: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "case_id": case_id,
            "external_id": external_id,
            "workflow_id": str(workflow_id) if workflow_id else None,
            "document_id": document_id,
            "hash": normalized.checksum,
            "content_hash": normalized.checksum,
            "source": normalized.document.source or "crawler",
        }

        collection_id = getattr(normalized.document.ref, "collection_id", None)
        if collection_id is not None:
            meta["collection_id"] = str(collection_id)

        title = getattr(normalized.document.meta, "title", None)
        if isinstance(title, str) and title.strip():
            meta["title"] = title.strip()

        embedding_profile = profile_binding.profile_id
        if embedding_profile:
            meta["embedding_profile"] = embedding_profile
        vector_space = profile_binding.resolution.vector_space
        if getattr(vector_space, "id", None):
            meta["vector_space_id"] = vector_space.id
        if getattr(vector_space, "dimension", None):
            meta["vector_space_dimension"] = str(vector_space.dimension)

        key_alias = embedding_state.get("key_alias")
        if isinstance(key_alias, str) and key_alias.strip():
            meta["key_alias"] = key_alias.strip()

        frontier_state = state.get("frontier")
        frontier_mapping: Optional[Mapping[str, Any]]
        if isinstance(frontier_state, FrontierData):
            frontier_mapping = frontier_state.model_dump()
        elif isinstance(frontier_state, Mapping):
            frontier_mapping = frontier_state
        else:
            frontier_mapping = None
        if frontier_mapping is not None:
            breadcrumbs = frontier_mapping.get("breadcrumbs")
            if isinstance(breadcrumbs, Iterable) and not isinstance(
                breadcrumbs, (str, bytes)
            ):
                trail = [str(part).strip() for part in breadcrumbs if str(part).strip()]
                if trail:
                    meta["breadcrumbs"] = trail

        return {key: value for key, value in meta.items() if value not in (None, "")}

    def _build_parsed_blocks_payload(
        self, parse_artifact: Any, fallback_text: str
    ) -> Optional[Dict[str, Any]]:
        if parse_artifact is None:
            return None

        if isinstance(parse_artifact, Mapping):
            text_blocks = parse_artifact.get("text_blocks")
            stats = parse_artifact.get("statistics")
        else:
            text_blocks = getattr(parse_artifact, "text_blocks", None)
            stats = getattr(parse_artifact, "statistics", None)

        if not text_blocks:
            return None
        blocks: list[Dict[str, Any]] = []
        for index, block in enumerate(text_blocks):
            if isinstance(block, Mapping):
                payload = dict(block)
            else:
                try:
                    payload = dict(block.__dict__)
                except Exception:
                    continue
            payload.setdefault("index", index)
            blocks.append(payload)
        if not blocks:
            return None

        statistics = dict(stats) if isinstance(stats, Mapping) else {}
        return {"blocks": blocks, "statistics": statistics, "text": fallback_text}

    def _persist_chunk_inputs(
        self,
        state: Dict[str, Any],
        chunk_meta: Dict[str, Any],
        normalized: NormalizedDocumentPayload,
        text: str,
    ) -> Tuple[str, Optional[str]]:
        tenant_segment = object_store.sanitize_identifier(str(chunk_meta["tenant_id"]))
        case_segment = object_store.sanitize_identifier(str(chunk_meta["case_id"]))
        seed = (
            chunk_meta.get("external_id")
            or chunk_meta.get("document_id")
            or normalized.checksum
        )
        try:
            filename = object_store.safe_filename(f"{seed}.txt")
        except Exception:
            filename = object_store.safe_filename(f"chunk-{uuid4()}.txt")
        text_path = "/".join([tenant_segment, case_segment, "text", filename])
        object_store.write_bytes(text_path, text.encode("utf-8"))

        artifacts = self._artifacts(state)
        artifacts["chunk_text_path"] = text_path

        parse_artifact = artifacts.get("parse_artifact")
        blocks_payload = self._build_parsed_blocks_payload(parse_artifact, text)
        blocks_path: Optional[str] = None
        if blocks_payload:
            blocks_filename = filename.rsplit(".", 1)[0] + ".parsed.json"
            blocks_path = "/".join(
                [tenant_segment, case_segment, "text", blocks_filename]
            )
            object_store.write_json(blocks_path, blocks_payload)
            chunk_meta["parsed_blocks_path"] = blocks_path
            artifacts["chunk_blocks_path"] = blocks_path

        return text_path, blocks_path

    def _chunk_document(
        self,
        state: Dict[str, Any],
        normalized: NormalizedDocumentPayload,
        embedding_state: Mapping[str, Any],
    ) -> Optional[ChunkComputation]:
        text = normalized.content_normalized or normalized.primary_text or ""
        if not str(text or "").strip():
            return None

        profile_binding = self._resolve_embedding_profile_binding(embedding_state)
        chunk_meta = self._prepare_chunk_meta(
            state, normalized, embedding_state, profile_binding
        )
        text_path, blocks_path = self._persist_chunk_inputs(
            state, chunk_meta, normalized, text
        )

        ingestion_tasks = self._ingestion_tasks()
        chunk_result = ingestion_tasks.chunk(chunk_meta, text_path)
        if not isinstance(chunk_result, Mapping):
            artifacts = self._artifacts(state)
            artifacts.setdefault("chunk_errors", []).append(
                {"error": "chunk_result_invalid", "result": chunk_result}
            )
            self._annotate_span(
                state,
                phase="chunk",
                extra={"error": "chunk_result_invalid"},
            )
            return None

        chunk_path = chunk_result.get("path")
        if not chunk_path:
            artifacts = self._artifacts(state)
            artifacts.setdefault("chunk_errors", []).append(
                {"error": "chunk_path_missing", "result": dict(chunk_result)}
            )
            self._annotate_span(
                state,
                phase="chunk",
                extra={"error": "chunk_path_missing"},
            )
            return None
        chunk_path_str = str(chunk_path)

        artifacts = self._artifacts(state)
        artifacts["chunks_path"] = chunk_path_str
        artifacts["chunk_meta"] = dict(chunk_meta)

        chunk_count: Optional[int] = None
        try:
            payload = object_store.read_json(chunk_path_str)
        except Exception:
            payload = None
            artifacts.setdefault("chunk_errors", []).append(
                {"error": "chunk_payload_read_failed", "path": chunk_path_str}
            )
            self._annotate_span(
                state,
                phase="chunk",
                extra={
                    "error": "chunk_payload_read_failed",
                    "chunk.path": chunk_path_str,
                },
            )
        if isinstance(payload, dict):
            raw_chunks = payload.get("chunks")
            if isinstance(raw_chunks, list):
                chunk_count = len(raw_chunks)
            parents_payload = payload.get("parents")
            if isinstance(parents_payload, dict) and parents_payload:
                artifacts["chunk_parents"] = parents_payload
        elif isinstance(payload, list):
            chunk_count = len(payload)
        if chunk_count is not None:
            artifacts["chunk_count"] = chunk_count
            self._annotate_span(
                state, phase="chunk", extra={"chunk.count": chunk_count}
            )

        return ChunkComputation(
            meta=chunk_meta,
            chunks_path=chunk_path_str,
            text_path=text_path,
            profile=profile_binding,
            blocks_path=blocks_path,
            chunk_count=chunk_count,
        )

    def _embed_with_chunks(
        self,
        state: Dict[str, Any],
        normalized: NormalizedDocumentPayload,
        embedding_state: Mapping[str, Any],
        chunk_info: ChunkComputation,
    ) -> EmbeddingResult:
        tenant_schema_value = (
            embedding_state.get("tenant_schema")
            if isinstance(embedding_state, Mapping)
            else None
        )
        scope = ScopeContext(
            tenant_id=str(chunk_info.meta["tenant_id"]),
            trace_id=str(state.get("trace_id") or uuid4()),
            invocation_id=str(uuid4()),
            ingestion_run_id=str(state.get("ingestion_run_id") or uuid4()),
            case_id=(
                str(chunk_info.meta.get("case_id"))
                if chunk_info.meta.get("case_id")
                else None
            ),
            tenant_schema=str(tenant_schema_value) if tenant_schema_value else None,
        )

        ingestion_tasks = self._ingestion_tasks()

        def _dispatch_ingestion(
            document_id: UUID,
            collection_ids: Sequence[UUID],
            embedding_profile: str | None,
            scope_value: str | None,
        ) -> None:
            dispatch_meta = dict(chunk_info.meta)
            dispatch_meta["document_id"] = str(document_id)
            if collection_ids:
                dispatch_meta["collection_id"] = str(collection_ids[0])
            if embedding_profile and not dispatch_meta.get("embedding_profile"):
                dispatch_meta["embedding_profile"] = embedding_profile
            if scope_value and not dispatch_meta.get("scope"):
                dispatch_meta["scope"] = scope_value

            upsert_kwargs: dict[str, Any] = {}
            if embedding_state:
                tenant_schema_dispatch = embedding_state.get("tenant_schema")
                if tenant_schema_dispatch is not None:
                    upsert_kwargs["tenant_schema"] = tenant_schema_dispatch
                vector_client = embedding_state.get("client")
                if vector_client is not None:
                    upsert_kwargs["vector_client"] = vector_client
                vector_factory = embedding_state.get("client_factory")
                if vector_factory is not None:
                    upsert_kwargs["vector_client_factory"] = vector_factory

            embed_result = ingestion_tasks.embed(dispatch_meta, chunk_info.chunks_path)
            embeddings_path = str(embed_result.get("path"))
            ingestion_tasks.upsert(dispatch_meta, embeddings_path, **upsert_kwargs)

        ingestion_result = ingest_document(
            scope,
            meta=chunk_info.meta,
            chunks_path=chunk_info.chunks_path,
            embedding_state=embedding_state,
            dispatcher=_dispatch_ingestion,
        )
        artifacts = self._artifacts(state)
        artifacts["document_id"] = ingestion_result.get("document_id")
        artifacts["collection_ids"] = ingestion_result.get("collection_ids")
        inserted = ingestion_result.get("status") == "queued"

        workflow_id_value = chunk_info.meta.get("workflow_id")
        workflow_id = None
        if workflow_id_value not in (None, ""):
            workflow_id = str(workflow_id_value)

        collection_value = None
        collection_ids = ingestion_result.get("collection_ids")
        if isinstance(collection_ids, list) and collection_ids:
            collection_value = collection_ids[0]
        elif chunk_info.meta.get("collection_id"):
            collection_value = chunk_info.meta.get("collection_id")
        collection_id = None
        if collection_value not in (None, ""):
            collection_id = str(collection_value)

        document_value = (
            ingestion_result.get("document_id")
            or chunk_info.meta.get("document_id")
            or normalized.document_id
        )

        chunk_meta_model = ChunkMeta(
            tenant_id=str(chunk_info.meta["tenant_id"]),
            case_id=str(chunk_info.meta["case_id"]),
            source=normalized.document.source or "crawler",
            hash=str(chunk_info.meta.get("hash") or normalized.checksum),
            external_id=str(chunk_info.meta["external_id"]),
            content_hash=str(
                chunk_info.meta.get("content_hash") or normalized.checksum
            ),
            embedding_profile=chunk_info.profile.profile_id,
            vector_space_id=chunk_info.profile.resolution.vector_space.id,
            workflow_id=workflow_id,
            collection_id=collection_id,
            document_id=str(document_value),
            lifecycle_state=normalized.document.lifecycle_state,
        )

        return EmbeddingResult(
            status="queued" if inserted else "skipped",
            chunks_inserted=int(inserted),
            embedding_profile=chunk_info.profile.profile_id,
            vector_space_id=chunk_info.profile.resolution.vector_space.id,
            chunk_meta=chunk_meta_model,
        )

    def _transition_metadata(self, state: Dict[str, Any]) -> Dict[str, str]:
        base_metadata = self._collect_span_metadata(state)
        metadata: Dict[str, str] = {}

        trace_candidate = base_metadata.get("trace_id")
        if not trace_candidate:
            trace_candidate = state.get("trace_id")
        if isinstance(trace_candidate, str):
            trace_candidate = trace_candidate.strip()
        if trace_candidate:
            metadata["trace_id"] = str(trace_candidate)

        workflow_candidate = base_metadata.get("workflow_id")
        if not workflow_candidate:
            workflow_candidate = state.get("workflow_id")
        if isinstance(workflow_candidate, str):
            workflow_candidate = workflow_candidate.strip()
        if workflow_candidate:
            metadata["workflow_id"] = str(workflow_candidate)

        document_candidate = base_metadata.get("document_id")
        if not document_candidate:
            document_candidate = state.get("document_id")
        if document_candidate:
            metadata["document_id"] = str(document_candidate)

        return metadata

    def _emit(
        self,
        name: str,
        transition: GraphTransition,
        run_id: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._event_emitter is not None:
            try:
                payload: Dict[str, Any] = {
                    "transition": transition.to_dict(),
                    "run_id": run_id,
                }
                if context:
                    payload.update(dict(context))
                self._event_emitter(name, payload)
            except Exception:  # pragma: no cover - defensive best effort
                pass

    @staticmethod
    def _decode_payload_text(payload: bytes) -> str:
        if not payload:
            return ""
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return payload.decode("latin-1", errors="ignore")

    @staticmethod
    def _should_prefetch_parse(document: NormalizedDocument, raw_text: str) -> bool:
        if not raw_text or not raw_text.strip():
            return False
        blob = getattr(document, "blob", None)
        media_type = getattr(blob, "media_type", "") or ""
        media_type = media_type.strip().lower()
        return media_type in {"text/html", "application/xhtml+xml"}

    @observe_span(name="crawler.ingestion.run")
    def start_crawl(self, state: StateMapping) -> Dict[str, Any]:
        """Legacy bootstrap hook retained for callers expecting the older API."""
        working_state: Dict[str, Any] = dict(state)
        control = working_state.get("control")
        if isinstance(control, Mapping):
            working_state["control"] = dict(control)
        elif control is None:
            working_state["control"] = {}
        else:
            try:
                working_state["control"] = dict(control)  # type: ignore[arg-type]
            except Exception:
                working_state["control"] = {}

        artifacts = working_state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            working_state["artifacts"] = artifacts
        working_state.setdefault("transitions", {})

        try:
            self._ensure_normalized_payload(working_state)
        except Exception:
            artifacts.setdefault(
                "failure", {"decision": "error", "reason": "start_crawl_failed"}
            )
            raise

        return working_state

    def run(
        self,
        state: StateMapping,
        meta: StateMapping | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        working_state: Dict[str, Any] = dict(state)
        meta_payload = dict(meta or {})
        working_state["meta"] = meta_payload
        working_state.setdefault("artifacts", {})
        working_state.setdefault("transitions", {})
        run_id = working_state.setdefault("graph_run_id", str(uuid4()))
        if not working_state.get("ingestion_run_id"):
            working_state["ingestion_run_id"] = meta_payload.get(
                "ingestion_run_id"
            ) or str(uuid4())

        artifacts = working_state.setdefault("artifacts", {})
        try:
            normalized_payload = self._ensure_normalized_payload(working_state)
        except Exception as exc:
            artifacts.setdefault("failure", {"decision": "error", "reason": str(exc)})
            raise

        working_state["content_hash"] = normalized_payload.checksum
        self._annotate_span(working_state, phase="run")

        nodes = (
            GraphNode("update_status_normalized", self._run_update_status),
            GraphNode("enforce_guardrails", self._run_guardrails),
            GraphNode("document_pipeline", self._run_document_pipeline),
            GraphNode("ingest_decision", self._run_ingest_decision),
            GraphNode("ingest", self._run_ingestion),
        )

        transitions: Dict[str, GraphTransition] = {}
        continue_flow = True
        last_transition: Optional[GraphTransition] = None

        for node in nodes:
            import sys

            sys.stderr.write(
                f"DEBUG: Executing node {node.name}, continue_flow={continue_flow}\n"
            )
            sys.stderr.flush()
            if not continue_flow:
                break
            try:
                transition, continue_flow = node.execute(working_state)
            except Exception as exc:  # pragma: no cover - orchestrated error path
                transition = self._handle_node_error(working_state, node, exc)
                continue_flow = False
            transition = self._with_transition_metadata(transition, working_state)
            transitions[node.name] = transition
            last_transition = transition
            self._emit(node.name, transition, run_id)

            control = working_state.get("control")
            if (
                isinstance(control, Mapping)
                and control.get("mode") == "fetch_only"
                and node.name == "document_pipeline"
            ):
                continue_flow = False

        finish_transition, _ = self._run_finish(working_state)
        finish_transition = self._with_transition_metadata(
            finish_transition, working_state
        )
        transitions["finish"] = finish_transition
        self._emit("finish", finish_transition, run_id)

        working_state["transitions"] = {
            name: payload.result.model_dump() for name, payload in transitions.items()
        }
        summary = finish_transition if finish_transition else last_transition
        result: Dict[str, Any] = {
            "decision": summary.decision if summary else None,
            "reason": summary.reason if summary else None,
            "severity": summary.severity if summary else None,
            "graph_run_id": run_id,
            "transitions": {
                name: payload.result.model_dump()
                for name, payload in transitions.items()
            },
        }
        if summary is not None:
            result["phase"] = summary.phase
            result["context"] = dict(summary.context)
        working_state["result"] = result
        self._annotate_span(
            working_state,
            phase="run",
            transition=summary,
            extra={"graph_run_id": run_id},
        )
        self.start_crawl(working_state)

        return working_state, result

    def _require(self, state: Dict[str, Any], key: str) -> Any:
        if key not in state:
            raise KeyError(f"state_missing_{key}")
        return state[key]

    def _artifacts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            state["artifacts"] = artifacts
        return artifacts

    def _ensure_normalized_payload(
        self, state: Dict[str, Any]
    ) -> NormalizedDocumentPayload:
        artifacts = self._artifacts(state)
        existing = artifacts.get("normalized_document")
        if isinstance(existing, NormalizedDocumentPayload):
            return existing

        normalized_input = state.get("normalized_document_input")
        if isinstance(normalized_input, Mapping):
            try:
                normalized_input = NormalizedDocument.model_validate(normalized_input)
            except ValidationError as exc:
                raise KeyError("normalized_document_input_invalid") from exc
            # Do NOT write the Pydantic object back to state to maintain JSON serializability
        if not isinstance(normalized_input, NormalizedDocument):
            raise KeyError("normalized_document_input_missing")

        review_value = str(
            state.get("review") or state.get("control", {}).get("review") or ""
        )
        review_value = review_value.strip().lower()
        if review_value == "required":
            tags = list(normalized_input.meta.tags or [])
            if "pending_review" not in tags:
                tags.append("pending_review")
                meta_copy = normalized_input.meta.model_copy(
                    update={"tags": tags}, deep=True
                )
                normalized_input = normalized_input.model_copy(
                    update={"meta": meta_copy}, deep=True
                )

        payload_bytes = document_payload_bytes(normalized_input)
        raw_text = self._decode_payload_text(payload_bytes)
        primary_text = normalized_primary_text(raw_text)

        parse_result: ParsedResult | None = None
        if self._should_prefetch_parse(normalized_input, raw_text):
            try:
                parse_result = self._parser_dispatcher.parse(
                    normalized_input, self._pipeline_config
                )
            except Exception:
                parse_result = None
            else:
                serialized_blocks = [
                    block.text.strip()
                    for block in parse_result.text_blocks
                    if getattr(block, "text", "").strip()
                ]
                if serialized_blocks:
                    primary_text = normalized_primary_text(
                        "\n\n".join(serialized_blocks)
                    )
        if parse_result is not None:
            artifacts.setdefault("prefetched_parse_result", parse_result)

        metadata_payload: Dict[str, Any] = {
            "tenant_id": normalized_input.meta.tenant_id,
            "workflow_id": normalized_input.meta.workflow_id,
            "case_id": state.get("case_id"),
            "source": normalized_input.source,
        }
        metadata = MappingProxyType(
            {key: value for key, value in metadata_payload.items() if value is not None}
        )

        payload = NormalizedDocumentPayload(
            document=normalized_input,
            primary_text=primary_text,
            payload_bytes=payload_bytes,
            metadata=metadata,
            content_raw=raw_text,
            content_normalized=primary_text,
        )
        artifacts["normalized_document"] = payload
        # Serialize to maintain JSON compatibility for Celery task payloads
        state["normalized_document_input"] = normalized_input.model_dump(mode="json")

        if "repository_baseline" not in artifacts:
            baseline = self._load_repository_baseline(state, payload)
            if baseline:
                artifacts["repository_baseline"] = baseline

        return payload

    def _load_repository_baseline(
        self, state: Dict[str, Any], normalized: NormalizedDocumentPayload
    ) -> Dict[str, Any]:
        state["_baseline_lookup_attempted"] = True
        repository = self._repository
        if repository is None:
            return {}
        try:
            existing = repository.get(
                normalized.tenant_id,
                normalized.document.ref.document_id,
                prefer_latest=True,
                workflow_id=normalized.document.ref.workflow_id,
            )
        except (AttributeError, NotImplementedError):
            return {}
        except Exception:
            return {}
        if existing is None:
            return {}

        baseline: Dict[str, Any] = {}
        checksum = getattr(existing, "checksum", None)
        if checksum:
            baseline.setdefault("checksum", checksum)
            baseline.setdefault("content_hash", checksum)
        ref = getattr(existing, "ref", None)
        if ref is not None:
            document_id = getattr(ref, "document_id", None)
            if document_id is not None:
                baseline.setdefault("document_id", str(document_id))
            collection_id = getattr(ref, "collection_id", None)
            if collection_id is not None:
                baseline.setdefault("collection_id", str(collection_id))
            version = getattr(ref, "version", None)
            if version:
                baseline.setdefault("version", version)
        lifecycle_state = getattr(existing, "lifecycle_state", None)
        if lifecycle_state:
            lifecycle_text = str(lifecycle_state)
            baseline.setdefault("lifecycle_state", lifecycle_text)
            state.setdefault("previous_status", lifecycle_text)
        return baseline

    @observe_span(name="crawler.ingestion.update_status")
    def _run_update_status(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        self._annotate_span(state, phase="update_status")
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts["normalized_document"]
        status = self._document_service.update_lifecycle_status(
            tenant_id=normalized.tenant_id,
            document_id=normalized.document_id,
            status="normalized",
            previous_status=state.get("previous_status"),
            workflow_id=normalized.document.ref.workflow_id,
            reason="document_normalized",
        )
        artifacts["status_update"] = status
        self._annotate_span(
            state,
            phase="update_status",
            extra={"status": getattr(status, "status", None)},
        )
        transition = _transition(
            phase="update_status",
            decision="status_updated",
            reason="lifecycle_normalized",
            severity="info",
            lifecycle=build_lifecycle_section(status),
            context={"lifecycle": dict(status.to_dict())},
        )
        return transition, True

    @staticmethod
    def _timedelta_from_ms(value: float | None) -> Optional[timedelta]:
        if value is None:
            return None
        if value <= 0:
            return None
        return timedelta(milliseconds=float(value))

    @staticmethod
    def _quota_limits_from_payload(
        max_docs: int | None, max_bytes: int | None
    ) -> Optional[QuotaLimits]:
        if max_docs is None and max_bytes is None:
            return None
        return QuotaLimits(max_documents=max_docs, max_bytes=max_bytes)

    @staticmethod
    def _quota_usage_from_payload(
        documents: int | None, bytes_used: int | None
    ) -> Optional[QuotaUsage]:
        if documents is None and bytes_used is None:
            return None
        return QuotaUsage(documents=int(documents or 0), bytes=int(bytes_used or 0))

    def _guardrail_limits_from_payload(
        self, payload: Optional[GuardrailStatePayload]
    ) -> Optional[GuardrailLimits]:
        if payload is None or payload.limits is None:
            return None
        limits = payload.limits
        return GuardrailLimits(
            max_document_bytes=limits.max_document_bytes,
            processing_time_limit=self._timedelta_from_ms(
                limits.processing_time_limit_ms
            ),
            mime_blacklist=frozenset(limits.mime_blacklist),
            host_blocklist=frozenset(limits.host_blocklist),
            tenant_quota=self._quota_limits_from_payload(
                limits.tenant_quota_max_docs, limits.tenant_quota_max_bytes
            ),
            host_quota=self._quota_limits_from_payload(
                limits.host_quota_max_docs, limits.host_quota_max_bytes
            ),
        )

    def _guardrail_signals_from_payload(
        self, payload: Optional[GuardrailStatePayload]
    ) -> Optional[GuardrailSignals]:
        if payload is None or payload.signals is None:
            return None
        signals = payload.signals
        return GuardrailSignals(
            tenant_id=signals.tenant_id,
            provider=signals.provider,
            canonical_source=signals.canonical_source,
            host=signals.host,
            document_bytes=signals.document_bytes,
            processing_time=self._timedelta_from_ms(signals.processing_time_ms),
            mime_type=signals.mime_type,
            tenant_usage=self._quota_usage_from_payload(
                signals.tenant_usage_docs, signals.tenant_usage_bytes
            ),
            host_usage=self._quota_usage_from_payload(
                signals.host_usage_docs, signals.host_usage_bytes
            ),
        )

    @staticmethod
    def _limits_data_from_dataclass(
        limits: Optional[GuardrailLimits],
    ) -> Optional[GuardrailLimitsData]:
        if limits is None:
            return None
        return GuardrailLimitsData(
            max_document_bytes=limits.max_document_bytes,
            processing_time_limit_ms=(
                limits.processing_time_limit.total_seconds() * 1000
                if limits.processing_time_limit
                else None
            ),
            mime_blacklist=tuple(limits.mime_blacklist),
            host_blocklist=tuple(limits.host_blocklist),
            tenant_quota_max_docs=(
                limits.tenant_quota.max_documents if limits.tenant_quota else None
            ),
            tenant_quota_max_bytes=(
                limits.tenant_quota.max_bytes if limits.tenant_quota else None
            ),
            host_quota_max_docs=(
                limits.host_quota.max_documents if limits.host_quota else None
            ),
            host_quota_max_bytes=(
                limits.host_quota.max_bytes if limits.host_quota else None
            ),
        )

    @staticmethod
    def _signals_data_from_dataclass(
        signals: Optional[GuardrailSignals],
    ) -> Optional[GuardrailSignalsData]:
        if signals is None or signals.tenant_id is None:
            return None
        return GuardrailSignalsData(
            tenant_id=str(signals.tenant_id),
            provider=signals.provider,
            canonical_source=signals.canonical_source,
            host=signals.host,
            document_bytes=signals.document_bytes or 0,
            mime_type=signals.mime_type,
            processing_time_ms=(
                signals.processing_time.total_seconds() * 1000
                if signals.processing_time
                else None
            ),
            tenant_usage_docs=(
                signals.tenant_usage.documents if signals.tenant_usage else 0
            ),
            tenant_usage_bytes=(
                signals.tenant_usage.bytes if signals.tenant_usage else 0
            ),
            host_usage_docs=(signals.host_usage.documents if signals.host_usage else 0),
            host_usage_bytes=(signals.host_usage.bytes if signals.host_usage else 0),
        )

    def _coerce_guardrail_payload(
        self,
        payload: Any,
    ) -> Optional[GuardrailStatePayload]:
        if isinstance(payload, GuardrailStatePayload):
            return payload
        if isinstance(payload, Mapping):
            legacy_limits = payload.get("limits")
            legacy_signals = payload.get("signals")
            if isinstance(legacy_limits, GuardrailLimits) or isinstance(
                legacy_signals, GuardrailSignals
            ):
                limits_data = self._limits_data_from_dataclass(
                    legacy_limits
                    if isinstance(legacy_limits, GuardrailLimits)
                    else None
                )
                signals_data = self._signals_data_from_dataclass(
                    legacy_signals
                    if isinstance(legacy_signals, GuardrailSignals)
                    else None
                )
                attributes = dict(payload.get("attributes", {}))
                config_override = payload.get("config")
                if isinstance(config_override, Mapping):
                    attributes.update(config_override)
                return GuardrailStatePayload(
                    decision=str(payload.get("decision", "allow")),
                    reason=str(payload.get("reason", "pending")),
                    allowed=bool(payload.get("allowed", True)),
                    policy_events=tuple(payload.get("policy_events", ())),
                    limits=limits_data,
                    signals=signals_data,
                    attributes=attributes,
                )
            try:
                return GuardrailStatePayload.model_validate(payload)
            except Exception:
                return None
        return None

    def _resolve_guardrail_state(self, state: Dict[str, Any]) -> Tuple[
        Optional[Mapping[str, Any]],
        Optional[GuardrailLimits],
        Optional[GuardrailSignals],
        Optional[Callable[..., Any]],
    ]:
        payload = self._coerce_guardrail_payload(state.get("guardrails"))
        if payload is None:
            return (None, None, None, None)
        return (
            payload.attributes or None,
            self._guardrail_limits_from_payload(payload),
            self._guardrail_signals_from_payload(payload),
            None,
        )

    def _resolve_frontier_state(
        self, state: Dict[str, Any]
    ) -> Optional[Mapping[str, Any]]:
        """Merge state and meta frontier payloads into a single mapping."""

        def _coerce_frontier(frontier: Any) -> Optional[Mapping[str, Any]]:
            if isinstance(frontier, FrontierData):
                return frontier.model_dump()
            if isinstance(frontier, Mapping):
                return dict(frontier)
            return None

        def _collect_policy_events(candidate: Any) -> Tuple[str, ...]:
            if candidate is None:
                return ()
            if isinstance(candidate, Mapping):
                maybe_events = candidate.get("policy_events")
                if maybe_events is candidate:
                    return ()
                return _collect_policy_events(maybe_events)
            if isinstance(candidate, str):
                value = candidate.strip()
                return (value,) if value else ()
            if isinstance(candidate, Iterable) and not isinstance(
                candidate, (bytes, bytearray)
            ):
                collected = []
                for item in candidate:
                    if not item:
                        continue
                    value = str(item).strip()
                    if value:
                        collected.append(value)
                return tuple(collected)
            value = str(candidate).strip()
            return (value,) if value else ()

        merged: Dict[str, Any] = {}
        policy_events: Tuple[str, ...] = ()

        def _merge_frontier(frontier: Mapping[str, Any]) -> None:
            nonlocal policy_events
            for key, value in frontier.items():
                if key == "policy_events":
                    events = _collect_policy_events(value)
                    if events:
                        policy_events = ai_core_api._merge_policy_events(
                            policy_events, events
                        )
                else:
                    merged[key] = value

        meta_payload = state.get("meta")
        if isinstance(meta_payload, Mapping):
            meta_frontier = _coerce_frontier(meta_payload.get("frontier"))
            if meta_frontier is not None:
                _merge_frontier(dict(meta_frontier))

        state_frontier = _coerce_frontier(state.get("frontier"))
        if state_frontier is not None:
            _merge_frontier(dict(state_frontier))

        if policy_events:
            merged["policy_events"] = list(policy_events)

        return merged or None

    @observe_span(name="crawler.ingestion.guardrails")
    def _run_guardrails(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        self._annotate_span(state, phase="guardrails")
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts["normalized_document"]
        config, limits, signals, error_builder = self._resolve_guardrail_state(state)
        frontier_state = self._resolve_frontier_state(state)
        decision = self._guardrail_enforcer(
            normalized_document=normalized,
            config=config,
            limits=limits,
            signals=signals,
            error_builder=error_builder,
            frontier_state=frontier_state,
        )
        artifacts["guardrail_decision"] = decision
        context: Dict[str, Any] = {
            key: value
            for key, value in decision.attributes.items()
            if key != "policy_events"
        }
        policy_events = tuple(decision.policy_events)
        context.setdefault("document_id", normalized.document_id)
        transition = _transition(
            phase="guardrails",
            decision=decision.decision,
            reason=decision.reason,
            severity="error" if not decision.allowed else "info",
            guardrail=build_guardrail_section(decision),
            context=context,
        )
        self._annotate_span(
            state,
            phase="guardrails",
            extra={"decision": decision.decision, "allowed": decision.allowed},
        )
        if not decision.allowed:
            reason_label = (decision.reason or "").strip() or "unknown"
            workflow_label = (
                normalized.document.ref.workflow_id or ""
            ).strip() or "unknown"
            tenant_label = (normalized.tenant_id or "").strip() or "unknown"
            source_label = (normalized.document.source or "").strip() or "unknown"
            document_metrics.GUARDRAIL_DENIAL_REASON_TOTAL.inc(
                reason=reason_label,
                workflow_id=workflow_label,
                tenant_id=tenant_label,
                source=source_label,
            )
            emit_event(
                "crawler_guardrail_denied",
                {
                    "reason": decision.reason,
                    "policy_events": list(decision.policy_events),
                },
            )
            status = self._document_service.update_lifecycle_status(
                tenant_id=normalized.tenant_id,
                document_id=normalized.document_id,
                status="deleted",
                workflow_id=normalized.document.ref.workflow_id,
                reason=decision.reason,
                policy_events=decision.policy_events,
            )
            artifacts.setdefault("status_updates", []).append(status)
            context_payload = {
                "document_id": normalized.document_id,
                "tenant_id": normalized.tenant_id,
                "reason": decision.reason,
                "policy_events": list(policy_events),
            }
            run_id = str(state.get("graph_run_id") or "")
            if run_id:
                self._emit(
                    "guardrail_denied",
                    transition,
                    run_id,
                    context=context_payload,
                )
        continue_flow = decision.allowed
        return transition, continue_flow

    @observe_span(name="crawler.ingestion.document_pipeline")
    def _run_document_pipeline(
        self, state: Dict[str, Any]
    ) -> Tuple[GraphTransition, bool]:
        import sys

        sys.stderr.write("DEBUG: _run_document_pipeline CALLED\n")
        sys.stderr.flush()
        # raise RuntimeError("DEBUG: SCREAM TEST - _run_document_pipeline REACHED")

        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts["normalized_document"]
        dry_run = bool(state.get("dry_run"))
        review_value = str(
            state.get("review") or state.get("control", {}).get("review") or ""
        )
        review_value = review_value.strip().lower()
        if dry_run:
            run_until = DocumentProcessingPhase.PARSE_ONLY
        elif review_value == "required":
            run_until = DocumentProcessingPhase.PARSE_AND_PERSIST
        else:
            run_until = DocumentProcessingPhase.FULL

        self._annotate_span(
            state,
            phase="document_pipeline",
            extra={"run_until": run_until.value},
        )

        case_id = state.get("case_id")
        trace_id = state.get("trace_id") or state.get("meta", {}).get("trace_id")
        span_id = state.get("span_id") or state.get("meta", {}).get("span_id")

        doc_collection_value = state.get("document_collection_id")
        document_collection_id = None
        if doc_collection_value:
            try:
                document_collection_id = UUID(str(doc_collection_value))
            except Exception:
                document_collection_id = None

        context = DocumentProcessingContext.from_document(
            normalized.document,
            case_id=str(case_id) if case_id else None,
            document_collection_id=document_collection_id,
            trace_id=str(trace_id) if trace_id else None,
            span_id=str(span_id) if span_id else None,
        )

        prefetched_parse = artifacts.get("prefetched_parse_result")
        if not isinstance(prefetched_parse, ParsedResult):
            prefetched_parse = None

        pipeline_state = DocumentProcessingState(
            document=normalized.document,
            config=self._pipeline_config,
            context=context,
            run_until=run_until,
            parsed_result=prefetched_parse,
        )

        baseline = artifacts.get("repository_baseline")
        if not isinstance(baseline, Mapping):
            baseline = self._load_repository_baseline(state, normalized)
            if baseline:
                artifacts["repository_baseline"] = baseline

        try:
            result_state = self._document_graph.invoke(pipeline_state)
        except Exception as exc:
            artifacts["document_pipeline_error"] = repr(exc)
            artifacts.setdefault(
                "failure",
                {"decision": "error", "reason": "document_pipeline_failed"},
            )
            self._annotate_span(
                state,
                phase="document_pipeline",
                extra={"error": repr(exc)},
            )
            raise

        artifacts["document_pipeline_phase"] = result_state.phase
        artifacts["document_processing_context"] = result_state.context

        def _serialize_artifact(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: _serialize_artifact(v) for k, v in asdict(obj).items()}
            if isinstance(obj, BaseModel):
                return obj.model_dump(mode="json")
            if isinstance(obj, (list, tuple)):
                return [_serialize_artifact(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _serialize_artifact(v) for k, v in obj.items()}
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        if result_state.parse_artifact:
            artifacts["parse_artifact"] = _serialize_artifact(
                result_state.parse_artifact
            )
        else:
            artifacts["parse_artifact"] = None

        # Ensure prefetched_parse_result is also serializable if present
        if "prefetched_parse_result" in artifacts:
            artifacts["prefetched_parse_result"] = _serialize_artifact(
                artifacts["prefetched_parse_result"]
            )

        # DEBUG LOGGING
        import sys

        sys.stderr.write(f"DEBUG: artifacts keys: {list(artifacts.keys())}\n")
        for k, v in artifacts.items():
            sys.stderr.write(f"DEBUG: artifacts[{k}] type: {type(v)}\n")
            if "ParsedResult" in str(type(v)):
                sys.stderr.write(f"DEBUG: FOUND ParsedResult in {k}!\n")
        sys.stderr.flush()

        artifacts["chunk_artifact"] = result_state.chunk_artifact
        if result_state.error is not None:
            artifacts["document_pipeline_error"] = repr(result_state.error)
        artifacts["document_pipeline_run_until"] = run_until.value

        updated_payload = NormalizedDocumentPayload(
            document=result_state.document,
            primary_text=normalized.content_normalized,
            payload_bytes=normalized.payload_bytes,
            metadata=normalized.metadata,
            content_raw=normalized.content_raw,
            content_normalized=normalized.content_normalized,
        )
        artifacts["normalized_document"] = updated_payload
        # Serialize to maintain JSON compatibility for Celery task payloads
        state["normalized_document_input"] = result_state.document.model_dump(
            mode="json"
        )

        extra = {"phase": result_state.phase}
        if result_state.error is not None:
            extra["error"] = repr(result_state.error)
        self._annotate_span(state, phase="document_pipeline", extra=extra)

        if result_state.error is not None:
            artifacts.setdefault(
                "failure",
                {"decision": "error", "reason": "document_pipeline_failed"},
            )
            transition = _transition(
                phase="document_pipeline",
                decision="error",
                reason="document_pipeline_failed",
                severity="error",
                pipeline=PipelineSection(
                    phase=result_state.phase,
                    run_until=run_until,
                    error=repr(result_state.error),
                ),
            )
            return transition, False

        transition = _transition(
            phase="document_pipeline",
            decision="processed",
            reason="document_pipeline_completed",
            severity="info",
            pipeline=PipelineSection(
                phase=result_state.phase,
                run_until=run_until,
            ),
        )
        return transition, True

    @observe_span(name="crawler.ingestion.ingest_decision")
    def _run_ingest_decision(
        self, state: Dict[str, Any]
    ) -> Tuple[GraphTransition, bool]:
        self._annotate_span(state, phase="ingest_decision")
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts["normalized_document"]

        baseline = artifacts.get("repository_baseline")
        if isinstance(baseline, Mapping):
            baseline_input = dict(baseline)
        else:
            baseline_input = {}

        existing_baseline = state.get("baseline")
        if isinstance(existing_baseline, Mapping):
            baseline_input.update(dict(existing_baseline))

        baseline_lookup_attempted = bool(state.get("_baseline_lookup_attempted"))

        should_reload_baseline = False
        if not baseline_input.get("checksum"):
            should_reload_baseline = not baseline_lookup_attempted
        elif not state.get("previous_status"):
            should_reload_baseline = True

        if should_reload_baseline:
            repository_baseline = self._load_repository_baseline(state, normalized)
            for key, value in repository_baseline.items():
                baseline_input.setdefault(key, value)
            if repository_baseline:
                artifacts["repository_baseline"] = repository_baseline

        state["baseline"] = baseline_input

        decision = self._delta_decider(
            normalized_document=normalized,
            baseline=baseline_input,
            frontier_state=self._resolve_frontier_state(state),
        )
        artifacts["delta_decision"] = decision

        status_update = self._document_service.update_lifecycle_status(
            tenant_id=normalized.tenant_id,
            document_id=normalized.document_id,
            status="active",
            workflow_id=normalized.document.ref.workflow_id,
            reason=decision.reason,
        )
        artifacts.setdefault("status_updates", []).append(status_update)

        ingest_action = "upsert" if decision.decision in {"new", "changed"} else "skip"
        control = state.get("control")
        if isinstance(control, Mapping) and control.get("mode") == "store_only":
            ingest_action = "skip"
        state["ingest_action"] = ingest_action

        transition = _transition(
            phase="ingest_decision",
            decision=decision.decision,
            reason=decision.reason,
            delta=decision,
            lifecycle=build_lifecycle_section(status_update),
        )
        self._annotate_span(
            state,
            phase="ingest_decision",
            extra={"decision": decision.decision},
        )
        return transition, True

    @observe_span(name="crawler.ingestion.ingest")
    def _run_ingestion(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        self._annotate_span(state, phase="ingest")
        artifacts = self._artifacts(state)
        normalized: Optional[NormalizedDocumentPayload] = artifacts.get(
            "normalized_document"
        )
        if not isinstance(normalized, NormalizedDocumentPayload):
            transition = _transition(
                phase="ingest",
                decision="skipped",
                reason="normalized_missing",
                severity="warn",
            )
            return transition, True

        delta: Optional[ai_core_api.DeltaDecision] = artifacts.get("delta_decision")
        if delta is None:
            self._annotate_span(
                state,
                phase="ingest",
                extra={"outcome": "skipped", "reason": "delta_missing"},
            )
            transition = _transition(
                phase="ingest",
                decision="skipped",
                reason="delta_missing",
                severity="warn",
            )
            return transition, True
        if delta.decision not in {"new", "changed"}:
            self._annotate_span(
                state,
                phase="ingest",
                extra={"outcome": "skipped", "delta": delta.decision},
            )
            transition = _transition(
                phase="ingest",
                decision="skipped",
                reason="delta_not_applicable",
                severity="info",
                delta=delta,
                context={"delta_decision": delta.decision},
            )
            return transition, True

        embedding_state = state.get("embedding") or {}
        chunk_info: Optional[ChunkComputation] = None
        try:
            chunk_info = self._chunk_document(state, normalized, embedding_state)
        except Exception as exc:  # pragma: no cover - defensive fallback
            artifacts.setdefault("chunk_errors", []).append(repr(exc))
            self._annotate_span(
                state,
                phase="chunk",
                extra={"error": repr(exc)},
            )

        handler = self._embedding_handler
        if chunk_info is not None and handler is ai_core_api.trigger_embedding:
            result = self._embed_with_chunks(
                state, normalized, embedding_state, chunk_info
            )
        else:
            result = handler(
                normalized_document=normalized,
                embedding_profile=embedding_state.get("profile"),
                tenant_id=normalized.tenant_id,
                case_id=state.get("case_id"),
                vector_client=embedding_state.get("client"),
                vector_client_factory=embedding_state.get("client_factory"),
            )
        artifacts["embedding_result"] = result
        self._annotate_span(
            state,
            phase="ingest",
            extra={"outcome": result.status},
        )

        pipeline_error = artifacts.get("document_pipeline_error")
        review_value = str(
            state.get("review") or state.get("control", {}).get("review") or ""
        ).strip()
        handler = getattr(self, "upsert_handler", None)
        if (
            callable(handler)
            and not state.get("dry_run")
            and not review_value
            and pipeline_error is None
        ):
            decision_payload = SimpleNamespace(
                attributes={"chunk_meta": result.chunk_meta},
                payload=result,
            )
            try:
                upsert_result = handler(decision_payload)
            except Exception as exc:  # pragma: no cover - defensive guard
                upsert_result = {"status": "error", "error": str(exc)}
            artifacts["upsert_result"] = upsert_result

        transition = _transition(
            phase="ingest",
            decision="embedding_triggered",
            reason="embedding_enqueued",
            severity="info",
            embedding=result,
            delta=delta,
        )
        return transition, True

    @observe_span(name="crawler.ingestion.finish")
    def _run_finish(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        artifacts = self._artifacts(state)
        normalized: Optional[NormalizedDocumentPayload] = artifacts.get(
            "normalized_document"
        )
        guardrail: Optional[ai_core_api.GuardrailDecision] = artifacts.get(
            "guardrail_decision"
        )
        delta: Optional[ai_core_api.DeltaDecision] = artifacts.get("delta_decision")
        embedding_result: Optional[EmbeddingResult] = artifacts.get("embedding_result")

        if normalized is None:
            transition = _transition(
                phase="finish",
                decision="error",
                reason="normalization_missing",
                severity="error",
            )
            return transition, False

        if guardrail is None:
            guardrail = ai_core_api.GuardrailDecision(
                decision="allow", reason="default", attributes={"severity": "info"}
            )
        if delta is None:
            delta = ai_core_api.DeltaDecision(
                decision="unknown",
                reason="delta_missing",
                attributes={"severity": "warn"},
            )

        payload = self._completion_builder(
            normalized_document=normalized,
            decision=delta,
            guardrails=guardrail,
            embedding_result=embedding_result,
        )
        if isinstance(payload, Mapping):
            payload = CompletionPayload.model_validate(payload)
        updates: Dict[str, Any] = {}
        pipeline_phase = artifacts.get("document_pipeline_phase")
        if pipeline_phase is not None:
            updates["pipeline_phase"] = pipeline_phase
        pipeline_run_until = artifacts.get("document_pipeline_run_until")
        if pipeline_run_until is not None:
            updates["pipeline_run_until"] = str(pipeline_run_until)
        pipeline_error = artifacts.get("document_pipeline_error")
        if pipeline_error is not None:
            updates["pipeline_error"] = str(pipeline_error)
        failure = artifacts.get("failure")
        raw_severity = guardrail.attributes.get("severity")
        severity = (
            str(raw_severity).strip().lower()
            if isinstance(raw_severity, str) and raw_severity.strip()
            else ("error" if not guardrail.allowed else "info")
        )
        decision_value = delta.decision
        reason_value = delta.reason
        if failure:
            severity = "error"
            decision_value = failure.get("decision", "error")
            reason_value = failure.get("reason", guardrail.reason)
            updates["failure"] = dict(failure)
        elif not guardrail.allowed:
            severity = "error"
            decision_value = "denied"
            reason_value = guardrail.reason
        if updates:
            payload = payload.model_copy(update=updates)
        transition = _transition(
            phase="finish",
            decision=decision_value,
            reason=reason_value,
            severity=severity,
            guardrail=guardrail,
            delta=delta,
            embedding=embedding_result,
            context={"result": payload.model_dump()},
        )
        self._annotate_span(
            state,
            phase="finish",
            transition=transition,
            extra={
                "severity": severity,
                "guardrail_decision": guardrail.decision if guardrail else None,
                "delta_decision": delta.decision if delta else None,
            },
        )
        state["summary"] = payload
        return transition, False

    def _handle_node_error(
        self, working_state: Dict[str, Any], node: GraphNode, exc: Exception
    ) -> GraphTransition:
        artifacts = self._artifacts(working_state)
        artifacts.setdefault("errors", []).append(
            {
                "node": node.name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        artifacts["failure"] = {"decision": "error", "reason": f"{node.name}_failed"}
        normalized = artifacts.get("normalized_document")
        if isinstance(normalized, NormalizedDocumentPayload):
            try:
                status: LifecycleStatusUpdate = (
                    self._document_service.update_lifecycle_status(
                        tenant_id=normalized.tenant_id,
                        document_id=normalized.document_id,
                        status="deleted",
                        workflow_id=normalized.document.ref.workflow_id,
                        reason=f"{node.name}_failed",
                    )
                )
                artifacts.setdefault("status_updates", []).append(status)
            except Exception:  # pragma: no cover - best effort
                pass
        phase_alias = {
            "update_status_normalized": "update_status",
            "enforce_guardrails": "guardrails",
            "document_pipeline": "document_pipeline",
            "ingest_decision": "ingest_decision",
            "ingest": "ingest",
        }
        transition = _transition(
            phase=phase_alias.get(node.name, "finish"),
            decision="error",
            reason=f"{node.name}_failed",
            severity="error",
            context={"error": repr(exc)},
        )
        return self._with_transition_metadata(transition, working_state)


GRAPH = CrawlerIngestionGraph(document_service=DocumentsApiLifecycleService())


def build_graph(
    *, event_emitter: Optional[Callable[[str, Mapping[str, Any]], None]] = None
) -> CrawlerIngestionGraph:
    if event_emitter is None:
        return GRAPH
    return CrawlerIngestionGraph(
        document_service=GRAPH._document_service,  # type: ignore[attr-defined]
        repository=GRAPH._repository,  # type: ignore[attr-defined]
        document_persistence=GRAPH._document_persistence,  # type: ignore[attr-defined]
        guardrail_enforcer=GRAPH._guardrail_enforcer,  # type: ignore[attr-defined]
        delta_decider=GRAPH._delta_decider,  # type: ignore[attr-defined]
        embedding_handler=GRAPH._embedding_handler,  # type: ignore[attr-defined]
        completion_builder=GRAPH._completion_builder,  # type: ignore[attr-defined]
        event_emitter=event_emitter,
    )


def run(
    state: StateMapping, meta: StateMapping | None = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return GRAPH.run(state, meta)


__all__ = ["CrawlerIngestionGraph", "GRAPH", "build_graph", "run", "GraphTransition"]
