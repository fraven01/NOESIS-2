"""LangGraph inspired orchestration for crawler ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple
from uuid import uuid4

from ai_core import api as ai_core_api
from ai_core.api import EmbeddingResult
from ai_core.infra.observability import emit_event, observe_span, update_observation
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
from ai_core.rag.guardrails import GuardrailLimits, GuardrailSignals

StateMapping = Mapping[str, Any] | MutableMapping[str, Any]


@dataclass(frozen=True)
class GraphTransition:
    """Standard transition payload returned by graph nodes."""

    decision: str
    reason: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.decision or "").strip():
            raise ValueError("decision_required")
        if not str(self.reason or "").strip():
            raise ValueError("reason_required")
        attributes = dict(self.attributes or {})
        attributes.setdefault("severity", "info")
        object.__setattr__(self, "attributes", attributes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class GraphNode:
    """Tie a node name to an execution callable."""

    name: str
    runner: Callable[[Dict[str, Any]], Tuple[GraphTransition, bool]]

    def execute(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        return self.runner(state)


def _transition(
    decision: str,
    reason: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
) -> GraphTransition:
    return GraphTransition(
        decision=decision, reason=reason, attributes=attributes or {}
    )


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
            storage_candidate = components.storage
            try:
                storage = storage_candidate()  # type: ignore[call-arg]
            except Exception:
                storage = storage_candidate
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
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        metadata = self._collect_span_metadata(state)
        metadata["phase"] = phase
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                metadata[key] = value
        if metadata:
            update_observation(metadata=metadata)

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
    def _should_prefetch_parse(
        document: NormalizedDocument, raw_text: str
    ) -> bool:
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
        working_state: Dict[str, Any] = dict(state)
        meta_payload = dict(meta or {})
        working_state["meta"] = meta_payload
        working_state.setdefault("artifacts", {})
        working_state.setdefault("transitions", {})
        run_id = working_state.setdefault("graph_run_id", str(uuid4()))

        artifacts = working_state.setdefault("artifacts", {})
        try:
            normalized_payload = self._ensure_normalized_payload(working_state)
        except Exception as exc:
            artifacts.setdefault("failure", {"decision": "error", "reason": str(exc)})
            raise

        working_state["content_hash"] = normalized_payload.checksum

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
            if not continue_flow:
                break
            try:
                transition, continue_flow = node.execute(working_state)
            except Exception as exc:  # pragma: no cover - orchestrated error path
                transition = self._handle_node_error(working_state, node, exc)
                continue_flow = False
            transitions[node.name] = transition
            last_transition = transition
            self._emit(node.name, transition, run_id)

        finish_transition, _ = self._run_finish(working_state)
        transitions["finish"] = finish_transition
        self._emit("finish", finish_transition, run_id)

        working_state["transitions"] = {
            name: payload.to_dict() for name, payload in transitions.items()
        }
        summary = finish_transition if finish_transition else last_transition
        result = {
            "decision": summary.decision if summary else None,
            "reason": summary.reason if summary else None,
            "attributes": dict(summary.attributes) if summary else {},
            "graph_run_id": run_id,
            "transitions": {
                name: payload.to_dict() for name, payload in transitions.items()
            },
        }
        working_state["result"] = result
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
            "request_id": state.get("request_id")
            or state.get("meta", {}).get("request_id"),
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
        state["normalized_document_input"] = normalized_input

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
            "status_updated",
            "lifecycle_normalized",
            attributes={"severity": "info", "result": status.to_dict()},
        )
        return transition, True

    def _resolve_guardrail_state(self, state: Dict[str, Any]) -> Tuple[
        Optional[Mapping[str, Any]],
        Optional[GuardrailLimits],
        Optional[GuardrailSignals],
        Optional[Callable[..., Any]],
    ]:
        guardrail_state = state.get("guardrails")
        config: Optional[Mapping[str, Any]] = None
        limits: Optional[GuardrailLimits] = None
        signals: Optional[GuardrailSignals] = None
        error_builder: Optional[Callable[..., Any]] = None

        if isinstance(guardrail_state, Mapping):
            maybe_limits = guardrail_state.get("limits")
            if isinstance(maybe_limits, GuardrailLimits):
                limits = maybe_limits
            maybe_signals = guardrail_state.get("signals")
            if isinstance(maybe_signals, GuardrailSignals):
                signals = maybe_signals
            maybe_builder = guardrail_state.get("error_builder")
            if callable(maybe_builder):
                error_builder = maybe_builder
            config_candidate = guardrail_state.get("config")
            if isinstance(config_candidate, Mapping):
                config = config_candidate
            elif limits is None and signals is None:
                config = guardrail_state
        elif isinstance(guardrail_state, GuardrailLimits):
            limits = guardrail_state
        elif isinstance(guardrail_state, GuardrailSignals):
            signals = guardrail_state

        return config, limits, signals, error_builder

    def _resolve_frontier_state(
        self, state: Dict[str, Any]
    ) -> Optional[Mapping[str, Any]]:
        """Merge state and meta frontier payloads into a single mapping."""

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
            meta_frontier = meta_payload.get("frontier")
            if isinstance(meta_frontier, Mapping):
                _merge_frontier(dict(meta_frontier))

        state_frontier = state.get("frontier")
        if isinstance(state_frontier, Mapping):
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
        attributes = dict(decision.attributes)
        policy_events = list(decision.policy_events)
        if policy_events:
            attributes.setdefault("policy_events", policy_events)
        if "severity" not in attributes:
            attributes["severity"] = "error" if not decision.allowed else "info"
        attributes.setdefault("document_id", normalized.document_id)
        transition = _transition(
            decision.decision,
            decision.reason,
            attributes=attributes,
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
                policy_events=decision.attributes.get("policy_events"),
            )
            artifacts.setdefault("status_updates", []).append(status)
            context_payload = {
                "document_id": normalized.document_id,
                "tenant_id": normalized.tenant_id,
                "reason": decision.reason,
                "policy_events": policy_events,
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

        context = DocumentProcessingContext.from_document(
            normalized.document,
            case_id=str(case_id) if case_id else None,
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
        artifacts["parse_artifact"] = result_state.parse_artifact
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
        state["normalized_document_input"] = result_state.document

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
                "error",
                "document_pipeline_failed",
                attributes={"severity": "error", "phase": result_state.phase},
            )
            return transition, False

        transition = _transition(
            "processed",
            "document_pipeline_completed",
            attributes={
                "severity": "info",
                "phase": result_state.phase,
                "run_until": run_until.value,
            },
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
        state["ingest_action"] = ingest_action

        transition = _transition(
            decision.decision,
            decision.reason,
            attributes=decision.attributes,
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
                "skipped",
                "normalized_missing",
                attributes={"severity": "warn"},
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
                "skipped",
                "delta_missing",
                attributes={"severity": "warn"},
            )
            return transition, True
        if delta.decision not in {"new", "changed"}:
            self._annotate_span(
                state,
                phase="ingest",
                extra={"outcome": "skipped", "delta": delta.decision},
            )
            transition = _transition(
                "skipped",
                "delta_not_applicable",
                attributes={"severity": "info", "delta": delta.decision},
            )
            return transition, True

        embedding_state = state.get("embedding") or {}
        result = self._embedding_handler(
            normalized_document=normalized,
            embedding_profile=embedding_state.get("profile"),
            tenant_id=normalized.tenant_id,
            case_id=state.get("case_id"),
            request_id=state.get("request_id"),
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
            "embedding_triggered",
            "embedding_enqueued",
            attributes={"severity": "info", "result": result.to_dict()},
        )
        return transition, True

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
                "error",
                "normalization_missing",
                attributes={"severity": "error"},
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
            embedding_result=embedding_result.to_dict() if embedding_result else None,
        )
        pipeline_phase = artifacts.get("document_pipeline_phase")
        if pipeline_phase is not None:
            payload["pipeline_phase"] = pipeline_phase
        pipeline_run_until = artifacts.get("document_pipeline_run_until")
        if pipeline_run_until is not None:
            payload["pipeline_run_until"] = pipeline_run_until
        pipeline_error = artifacts.get("document_pipeline_error")
        if pipeline_error is not None:
            payload["pipeline_error"] = str(pipeline_error)
        failure = artifacts.get("failure")
        severity = guardrail.attributes.get("severity", "info")
        decision_value = delta.decision
        reason_value = delta.reason
        if failure:
            severity = "error"
            decision_value = failure.get("decision", "error")
            reason_value = failure.get("reason", guardrail.reason)
            payload["failure"] = dict(failure)
        elif not guardrail.allowed:
            severity = "error"
            decision_value = "denied"
            reason_value = guardrail.reason
        transition = _transition(
            decision_value,
            reason_value,
            attributes={"severity": severity, "result": dict(payload)},
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
        return _transition(
            "error",
            f"{node.name}_failed",
            attributes={"severity": "error", "error": repr(exc)},
        )


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
