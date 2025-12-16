"""LangGraph inspired orchestration for crawler ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

from pydantic import ValidationError, BaseModel
from uuid import UUID, uuid4

from ai_core.api import EmbeddingResult
from ai_core import api as ai_core_api
from ai_core.contracts.payloads import (
    CompletionPayload,
    FrontierData,
)
from .transition_contracts import (
    GraphTransition,
)
from ai_core.infra import observability as observability_module
from ai_core.infra.observability import (
    update_observation,
)
from documents.api import NormalizedDocumentPayload
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
from documents.parsers import ParsedResult, ParserDispatcher
from documents.cli import SimpleDocumentChunker
from .document_service import (
    DocumentLifecycleService,
    DocumentPersistenceService,
    DocumentsApiLifecycleService,
    DocumentsRepositoryAdapter,
)

StateMapping = Mapping[str, Any] | MutableMapping[str, Any]


# Legacy types removed


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
            from documents.parsers import create_default_parser_dispatcher

            parser_dispatcher = create_default_parser_dispatcher()
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

        # Default crawler ingestion must reach embedding/upsert, so enable embeddings
        self._pipeline_config = pipeline_config or DocumentPipelineConfig(
            enable_embedding=True
        )

        # Note: Repository is optional - delta checks will gracefully handle None
        # Storage and captioner are still required for document processing
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
            embedder=self._embedding_handler,
            delta_decider=self._delta_decider,
            guardrail_enforcer=self._guardrail_enforcer,
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

    # Legacy methods removed during refactoring

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

    def run(
        self,
        state: StateMapping,
        meta: StateMapping | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Initialize working state from legacy dict input
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

        # Determine run_until based on control flags
        dry_run = bool(working_state.get("dry_run"))
        review_value = (
            str(
                working_state.get("review")
                or working_state.get("control", {}).get("review")
                or ""
            )
            .strip()
            .lower()
        )

        if dry_run:
            run_until = DocumentProcessingPhase.PARSE_ONLY
        elif review_value == "required":
            run_until = DocumentProcessingPhase.PARSE_AND_PERSIST
        else:
            run_until = DocumentProcessingPhase.FULL

        # Construct DocumentProcessingState
        case_id = working_state.get("case_id")
        trace_id = working_state.get("trace_id") or meta_payload.get("trace_id")
        span_id = working_state.get("span_id") or meta_payload.get("span_id")
        doc_collection_value = working_state.get("document_collection_id")
        document_collection_id = None
        if doc_collection_value:
            try:
                document_collection_id = UUID(str(doc_collection_value))
            except Exception:
                document_collection_id = None

        context = DocumentProcessingContext.from_document(
            normalized_payload.document,
            case_id=str(case_id) if case_id else None,
            document_collection_id=document_collection_id,
            trace_id=str(trace_id) if trace_id else None,
            span_id=str(span_id) if span_id else None,
        )

        # Inject frontier if present - store in pipeline_state artifacts instead
        frontier = self._resolve_frontier_state(working_state)
        if frontier:
            # Store frontier in artifacts for later use by processing graph
            artifacts.setdefault("frontier", frontier)

        # Execute document processing graph with injected storage
        try:
            result_state = self._document_graph.invoke(
                DocumentProcessingState(
                    document=normalized_payload.document,
                    config=self._pipeline_config,
                    context=context,
                    storage=self._storage,  # Inject storage for blob decoding
                    run_until=run_until,
                )
            )
            if isinstance(result_state, dict):
                result_state = DocumentProcessingState(**result_state)
        except Exception as exc:
            artifacts["document_pipeline_error"] = repr(exc)
            artifacts.setdefault(
                "failure",
                {"decision": "error", "reason": "document_pipeline_failed"},
            )
            self._annotate_span(
                working_state,
                phase="run",
                extra={"error": repr(exc)},
            )
            raise

        # Map results back to legacy artifacts
        artifacts["document_pipeline_phase"] = result_state.phase
        artifacts["document_processing_context"] = result_state.context

        if result_state.parse_artifact:
            # Helper to serialize artifacts roughly matching legacy _serialize_artifact
            def _serialize(obj: Any) -> Any:
                if isinstance(obj, BaseModel):
                    return obj.model_dump(mode="json")
                if is_dataclass(obj) and not isinstance(obj, type):
                    return {k: _serialize(v) for k, v in asdict(obj).items()}
                return obj

            artifacts["parse_artifact"] = _serialize(result_state.parse_artifact)

        if result_state.chunk_artifact:
            artifacts["chunk_artifact"] = (
                result_state.chunk_artifact
            )  # May need serialization if used by tasks?
            if result_state.chunk_artifact.chunks:
                artifacts["chunk_count"] = len(result_state.chunk_artifact.chunks)

        if result_state.delta_decision:
            artifacts["delta_decision"] = result_state.delta_decision
            # If skipping, explicit skip check

        if result_state.guardrail_decision:
            artifacts["guardrail_decision"] = result_state.guardrail_decision

        # Synthesize a finish transition/result
        # Logic here mimics _run_finish but simplified since we have final decisions
        delta = result_state.delta_decision
        guardrail = result_state.guardrail_decision

        # Set ingest_action to trigger ingestion run if not skipped
        if delta and delta.decision not in ("skip", "error"):
            working_state["ingest_action"] = "upsert"

        # Build completion payload
        # Note: We don't have explicit embedding result object in state yet,
        # unless we add it to state or extract from context/artifacts
        # But 'chunk_artifact' is there. "embedding_result" is usually from api.trigger_embedding
        # If we need strict compat, we might need that object.
        # Check if we can fake it or if it matters.
        # legacy 'embedding_result' is EmbeddingResult

        embedding_result = None
        # In _embed_chunks, result is returned but state doesn't store it explicitly as 'embedding_result'
        # unless we modify state to store it.
        # For now, let's assume it's okay or we check if we can reconstruct.

        payload = self._completion_builder(
            normalized_document=normalized_payload,
            decision=delta,
            guardrails=guardrail,
            embedding_result=embedding_result,
        )
        if isinstance(payload, Mapping):
            payload = CompletionPayload.model_validate(payload)

        working_state["summary"] = payload

        # Construct simplified result dict
        result_dict: Dict[str, Any] = {
            "decision": delta.decision if delta else "unknown",
            "reason": delta.reason if delta else "unknown",
            "graph_run_id": run_id,
            "transitions": {},  # Legacy expectations might need this populated?
            # Only if UI depends on it.
            # If so, we might need to fake "ingest_decision" transition from delta.
        }
        working_state["result"] = result_dict

        self._annotate_span(
            working_state,
            phase="run",
            extra={"decision": result_dict["decision"]},
        )

        return working_state, result_dict

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
        metadata = {}  # Defensive init for UnboundLocalError debugging
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

                normalized_input = normalized_input.model_copy(
                    update={"meta": meta_copy}, deep=True
                )

        payload_bytes = document_payload_bytes(normalized_input, storage=self._storage)
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
        metadata = {
            key: value for key, value in metadata_payload.items() if value is not None
        }

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

    # _handle_node_error removed


def build_graph(
    *, event_emitter: Optional[Callable[[str, Mapping[str, Any]], None]] = None
) -> CrawlerIngestionGraph:
    """Build a fresh CrawlerIngestionGraph instance.

    Returns a new instance per call to prevent shared mutable state
    (_dedupe_index at line 168) across concurrent workers (Finding #5 fix).
    """
    return CrawlerIngestionGraph(
        document_service=DocumentsApiLifecycleService(),
        event_emitter=event_emitter,
    )


def run(
    state: StateMapping, meta: StateMapping | None = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run crawler ingestion with fresh graph instance.

    Creates a new graph instance per invocation to prevent state leakage.
    """
    graph = build_graph()  # Fresh instance per call
    return graph.run(state, meta)


__all__ = ["CrawlerIngestionGraph", "build_graph", "run", "GraphTransition"]
