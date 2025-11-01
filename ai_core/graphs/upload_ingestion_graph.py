"""Upload ingestion graph orchestrating document processing for uploads."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping
from uuid import uuid4

from django.conf import settings


from ai_core import api as ai_core_api
from ai_core.infra.observability import emit_event, observe_span, update_observation
from documents.api import NormalizedDocumentPayload
from documents.contracts import DocumentMeta, DocumentRef, InlineBlob, NormalizedDocument


# Lightweight transition modelling -----------------------------------------


@dataclass(frozen=True)
class GraphTransition:
    """Represents a deterministic node transition."""

    decision: str
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "diagnostics": dict(self.diagnostics),
        }


def _make_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {
            f.name: _make_json_safe(getattr(value, f.name))
            for f in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            (str(k) if not isinstance(k, str) else k): _make_json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_make_json_safe(item) for item in value)
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, set):
        return tuple(
            _make_json_safe(item)
            for item in sorted(value, key=lambda candidate: repr(candidate))
        )
    return value


def _json_safe_mapping(mapping: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not mapping:
        return {}
    return {
        (str(key) if not isinstance(key, str) else key): _make_json_safe(value)
        for key, value in mapping.items()
    }


def _transition(decision: str, reason: str, *, diagnostics: Mapping[str, Any] | None = None) -> GraphTransition:
    payload = dict(diagnostics or {})
    payload.setdefault("severity", "info")
    return GraphTransition(decision=decision, reason=reason, diagnostics=payload)


# Feature flag defaults -----------------------------------------------------


DEFAULT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_MIME_ALLOWLIST: tuple[str, ...] = (
    "text/plain",
    "text/markdown",
    "text/html",
    "application/octet-stream",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


class UploadIngestionError(RuntimeError):
    """Raised for unexpected internal errors in the upload ingestion graph."""


class UploadIngestionGraph:
    """High level orchestration for processing uploaded documents."""

    _RUN_ORDER = (
        "accept_upload",
        "quarantine_scan",
        "deduplicate",
        "parse",
        "normalize",
        "delta_and_guardrails",
        "persist_document",
        "chunk_and_embed",
        "lifecycle_hook",
        "finalize",
    )

    _RUN_UNTIL_TO_NODE = {
        "upload_accepted": "accept_upload",
        "scan_complete": "quarantine_scan",
        "dedupe_complete": "deduplicate",
        "parse_complete": "parse",
        "normalize_complete": "normalize",
        "guardrail_complete": "delta_and_guardrails",
        "persist_complete": "persist_document",
        "chunk_complete": "chunk_and_embed",
        "vector_complete": "chunk_and_embed",
        "lifecycle_complete": "lifecycle_hook",
    }

    def __init__(
        self,
        *,
        quarantine_scanner: Callable[[bytes, Mapping[str, Any]], GraphTransition] | None = None,
        guardrail_enforcer: Callable[..., ai_core_api.GuardrailDecision]
        | None = ai_core_api.enforce_guardrails,
        delta_decider: Callable[..., ai_core_api.DeltaDecision]
        | None = ai_core_api.decide_delta,
        persistence_handler: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        embedding_handler: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        lifecycle_hook: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._quarantine_scanner = quarantine_scanner
        self._guardrail_enforcer = guardrail_enforcer
        self._delta_decider = delta_decider
        self._persist = persistence_handler
        self._embed = embedding_handler
        self._lifecycle_hook = lifecycle_hook
        self._dedupe_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        self._source_versions: dict[tuple[str, str, str], int] = {}

    # ------------------------------------------------------------------ API

    @observe_span(name="upload.ingestion.run")
    def run(self, payload: Mapping[str, Any], run_until: str | None = None) -> Mapping[str, Any]:
        """Execute the upload ingestion graph for *payload*."""

        state = self._initial_state(payload)
        self._annotate_span(state, phase="run")
        target_node = self._RUN_UNTIL_TO_NODE.get(run_until or "")
        telemetry = state["telemetry"]
        transitions: dict[str, Mapping[str, Any]] = {}

        for node_name in self._RUN_ORDER:
            transition = self._execute_node(node_name, state)
            transition = self._with_transition_metadata(transition, state)
            telemetry.setdefault("nodes", {})[node_name] = {
                "span": self._span_name(node_name)
            }
            transitions[node_name] = transition.to_dict()
            state["doc"]["last_transition"] = transition.to_dict()
            state["doc"]["decision"] = transition.decision
            state["doc"]["reason"] = transition.reason

            self._maybe_emit_transition_event(node_name, transition, state)

            if node_name == target_node:
                break

            if transition.decision.startswith("skip"):
                break

        telemetry["ended_at"] = datetime.now(timezone.utc).isoformat()
        telemetry["total_ms"] = self._compute_total_ms(telemetry)
        state["doc"]["transitions"] = transitions

        result = {
            "decision": state["doc"].get("decision"),
            "reason": state["doc"].get("reason"),
            "document_id": state["doc"].get("document_id"),
            "version": state["doc"].get("version"),
            "snippets": state["doc"].get("snippets", []),
            "warnings": state["doc"].get("warnings", []),
            "telemetry": telemetry,
            "prompt_version": state["meta"].get("prompt_version"),
            "transitions": transitions,
        }
        self._annotate_span(
            state,
            phase="run",
            extra={
                "decision": result.get("decision"),
                "reason": result.get("reason"),
                "document_id": state["doc"].get("document_id"),
            },
        )
        return result

    # -------------------------------------------------------------- Node impl

    def _execute_node(self, name: str, state: MutableMapping[str, Any]) -> GraphTransition:
        try:
            handler = getattr(self, f"_node_{name}")
        except AttributeError as exc:  # pragma: no cover - defensive
            raise UploadIngestionError(f"node_missing:{name}") from exc
        try:
            return handler(state)
        except UploadIngestionError as exc:
            self._handle_node_error(name, exc, state)
            raise
        except Exception as exc:
            self._handle_node_error(name, exc, state)
            raise UploadIngestionError(f"node_failed:{name}") from exc

    # Node: accept_upload ----------------------------------------------------

    @observe_span(name="upload.ingestion.accept_upload", auto_annotate=True)
    def _node_accept_upload(self, state: MutableMapping[str, Any]) -> GraphTransition:
        payload = state["input"]
        tenant_id = self._require_str(payload, "tenant_id")
        uploader_id = self._require_str(payload, "uploader_id")
        trace_id = self._require_str(payload, "trace_id")
        visibility = self._resolve_visibility(payload.get("visibility"))
        workflow_id = self._normalize_optional_str(payload.get("workflow_id")) or "upload"
        case_id = self._normalize_optional_str(payload.get("case_id"))

        file_bytes = payload.get("file_bytes")
        file_uri = payload.get("file_uri")
        if file_bytes and file_uri:
            return _transition(
                "skip_invalid_input",
                "multiple_sources",
                diagnostics={"severity": "error"},
            )
        if not file_bytes and not file_uri:
            return _transition(
                "skip_invalid_input",
                "payload_missing",
                diagnostics={"severity": "error"},
            )

        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")
        elif not isinstance(file_bytes, (bytes, bytearray)) and file_bytes is not None:
            return _transition(
                "skip_invalid_input",
                "bytes_required",
                diagnostics={"severity": "error"},
            )

        if file_bytes is None:
            raise UploadIngestionError("file_uri_not_supported")

        binary = bytes(file_bytes)
        size_limit = int(getattr(settings, "UPLOAD_MAX_BYTES", DEFAULT_MAX_BYTES))
        if len(binary) > size_limit:
            return _transition(
                "skip_oversize",
                "file_too_large",
                diagnostics={"max_bytes": size_limit, "size": len(binary), "severity": "warn"},
            )

        declared_mime = self._normalize_mime(payload.get("declared_mime"))
        filename = (payload.get("filename") or "upload").strip()
        if not declared_mime and filename:
            guess, _ = mimetypes.guess_type(filename)
            declared_mime = self._normalize_mime(guess)
        if not declared_mime:
            declared_mime = "application/octet-stream"

        allowed = getattr(settings, "UPLOAD_ALLOWED_MIME_TYPES", DEFAULT_MIME_ALLOWLIST)
        allowlist = tuple(str(item).strip().lower() for item in allowed)
        if allowlist and declared_mime not in allowlist:
            return _transition(
                "skip_disallowed_mime",
                "mime_not_allowed",
                diagnostics={"mime": declared_mime, "allowed": allowlist},
            )

        state["meta"].update(
            {
                "tenant_id": tenant_id,
                "uploader_id": uploader_id,
                "visibility": visibility,
                "source": "upload",
                "tags": tuple(self._normalize_tags(payload.get("tags"))),
                "source_key": self._normalize_optional_str(payload.get("source_key")),
                "filename": filename,
                "workflow_id": workflow_id,
                "trace_id": trace_id,
            }
        )
        if case_id:
            state["meta"]["case_id"] = case_id
        state["ingest"].update(
            {
                "size": len(binary),
                "declared_mime": declared_mime,
                "binary": binary,
            }
        )
        return _transition(
            "accepted",
            "input_valid",
            diagnostics={"size": len(binary), "visibility": visibility},
        )

    # Node: quarantine_scan --------------------------------------------------

    @observe_span(name="upload.ingestion.quarantine_scan", auto_annotate=True)
    def _node_quarantine_scan(self, state: MutableMapping[str, Any]) -> GraphTransition:
        enabled = getattr(settings, "UPLOAD_QUARANTINE_ENABLED", False)
        if not enabled or self._quarantine_scanner is None:
            return _transition("proceed", "quarantine_disabled")

        binary = state["ingest"].get("binary", b"")
        context = {
            "tenant_id": state["meta"].get("tenant_id"),
            "visibility": state["meta"].get("visibility"),
            "declared_mime": state["ingest"].get("declared_mime"),
        }
        transition = self._quarantine_scanner(binary, context)
        if not isinstance(transition, GraphTransition):
            raise UploadIngestionError("quarantine_scanner_invalid")
        return transition

    # Node: deduplicate ------------------------------------------------------

    @observe_span(name="upload.ingestion.deduplicate", auto_annotate=True)
    def _node_deduplicate(self, state: MutableMapping[str, Any]) -> GraphTransition:
        binary: bytes = state["ingest"].get("binary", b"")
        tenant_id = state["meta"].get("tenant_id")
        visibility = state["meta"].get("visibility")
        content_hash = hashlib.sha256(binary).hexdigest()
        state["ingest"]["content_hash"] = content_hash

        source = state["meta"].get("source") or "upload"
        dedupe_key = (tenant_id, source, content_hash)
        existing = self._dedupe_index.get(dedupe_key)
        if existing is not None:
            state["doc"].update(existing)
            return _transition(
                "skip_duplicate",
                "content_hash_seen",
                diagnostics={"severity": "info", "content_hash": content_hash},
            )

        source_key = state["meta"].get("source_key")
        if source_key:
            version_key = (tenant_id, source, source_key)
            version = self._source_versions.get(version_key, 0) + 1
            self._source_versions[version_key] = version
            state["doc"]["version"] = str(version)

        state["doc"]["content_hash"] = content_hash
        return _transition("proceed", "dedupe_ok", diagnostics={"content_hash": content_hash})

    # Node: parse ------------------------------------------------------------

    @observe_span(name="upload.ingestion.parse", auto_annotate=True)
    def _node_parse(self, state: MutableMapping[str, Any]) -> GraphTransition:
        binary: bytes = state["ingest"].get("binary", b"")
        charset = "utf-8"
        try:
            text = binary.decode(charset)
        except UnicodeDecodeError:
            text = binary.decode("latin-1", errors="ignore")

        state["doc"]["raw_text"] = text
        snippets = text.splitlines()
        if snippets:
            state["doc"]["snippets"] = snippets[:3]
        return _transition("parse_complete", "parse_succeeded", diagnostics={"length": len(text)})

    # Node: normalize --------------------------------------------------------

    @observe_span(name="upload.ingestion.normalize", auto_annotate=True)
    def _node_normalize(self, state: MutableMapping[str, Any]) -> GraphTransition:
        raw_text = state["doc"].get("raw_text", "")
        normalized = " ".join(raw_text.split()) if raw_text else ""
        state["doc"]["normalized_text"] = normalized
        return _transition("normalize_complete", "normalized", diagnostics={"length": len(normalized)})

    # Node: delta and guardrails --------------------------------------------

    @observe_span(name="upload.ingestion.delta_and_guardrails", auto_annotate=True)
    def _node_delta_and_guardrails(self, state: MutableMapping[str, Any]) -> GraphTransition:
        normalized = self._build_normalized_document(state)
        baseline = self._resolve_baseline(state)
        state["doc"]["normalized_document"] = normalized.to_dict()
        state["doc"]["baseline"] = baseline

        guardrail_transition_dict: Mapping[str, Any] | None = None
        if self._guardrail_enforcer is not None:
            guardrail_decision = self._guardrail_enforcer(
                normalized_document=normalized,
            )
            state["doc"]["guardrail_decision"] = {
                "decision": guardrail_decision.decision,
                "reason": guardrail_decision.reason,
                "attributes": _json_safe_mapping(guardrail_decision.attributes),
            }
            guardrail_transition = self._translate_guardrail_decision(
                guardrail_decision,
                normalized,
            )
            guardrail_transition_dict = guardrail_transition.to_dict()
            state["doc"]["guardrail_transition"] = guardrail_transition_dict
            if guardrail_transition.decision.startswith("skip"):
                return guardrail_transition

        if self._delta_decider is None:
            diagnostics: Dict[str, Any] = {}
            if guardrail_transition_dict is not None:
                diagnostics["guardrail"] = guardrail_transition_dict
            return _transition("upsert", "delta_skipped", diagnostics=diagnostics)

        delta_decision = self._delta_decider(
            normalized_document=normalized,
            baseline=baseline,
        )
        state["doc"]["delta_decision"] = {
            "decision": delta_decision.decision,
            "reason": delta_decision.reason,
            "attributes": _json_safe_mapping(delta_decision.attributes),
        }
        delta_transition = self._translate_delta_decision(
            delta_decision,
            normalized,
            guardrail_transition_dict,
        )
        state["doc"]["delta"] = delta_transition.to_dict()
        return delta_transition

    def _build_normalized_document(
        self, state: MutableMapping[str, Any]
    ) -> NormalizedDocumentPayload:
        tenant_id = state["meta"].get("tenant_id")
        if not tenant_id:
            raise UploadIngestionError("tenant_missing_for_guardrail")
        workflow_id = state["meta"].get("workflow_id") or "upload"
        filename = state["meta"].get("filename") or "upload"
        visibility = state["meta"].get("visibility")
        tags = list(state["meta"].get("tags") or ())
        uploader_id = state["meta"].get("uploader_id")
        source_key = state["meta"].get("source_key")
        case_id = state["meta"].get("case_id")
        trace_id = state["meta"].get("trace_id")

        binary: bytes = state["ingest"].get("binary", b"")
        declared_mime = state["ingest"].get("declared_mime") or "application/octet-stream"
        content_hash = state["doc"].get("content_hash")
        if not content_hash:
            raise UploadIngestionError("content_hash_missing_for_guardrail")
        normalized_text = state["doc"].get("normalized_text") or ""
        raw_text = state["doc"].get("raw_text") or normalized_text

        document_id_value = state["doc"].get("document_id") or uuid4()
        version = state["doc"].get("version")

        external_ref: dict[str, str] = {"provider": "upload"}
        if source_key:
            external_ref["external_id"] = str(source_key)
        if uploader_id:
            external_ref["uploader_id"] = str(uploader_id)

        meta = DocumentMeta(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            title=filename,
            tags=tags,
            origin_uri=self._normalize_optional_str(state["input"].get("origin_uri"))
            or self._normalize_optional_str(state["input"].get("file_uri")),
            external_ref=external_ref or None,
        )

        blob = InlineBlob(
            type="inline",
            media_type=declared_mime,
            base64=base64.b64encode(binary).decode("ascii"),
            sha256=content_hash,
            size=len(binary),
        )
        normalized_document = NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                document_id=document_id_value,
                version=version,
            ),
            meta=meta,
            blob=blob,
            checksum=content_hash,
            created_at=datetime.now(timezone.utc),
            source="upload",
        )

        metadata_payload = {
            "tenant_id": tenant_id,
            "workflow_id": workflow_id,
            "case_id": case_id,
            "trace_id": trace_id,
            "source": "upload",
            "visibility": visibility,
            "filename": filename,
        }

        metadata = MappingProxyType(
            {key: value for key, value in metadata_payload.items() if value is not None}
        )

        return NormalizedDocumentPayload(
            document=normalized_document,
            primary_text=normalized_text or raw_text,
            payload_bytes=binary,
            metadata=metadata,
            content_raw=raw_text,
            content_normalized=normalized_text or raw_text,
        )

    def _resolve_baseline(self, state: MutableMapping[str, Any]) -> Mapping[str, Any]:
        baseline_candidate = state["doc"].get("baseline")
        if not isinstance(baseline_candidate, Mapping):
            baseline_candidate = state["input"].get("baseline")
        if isinstance(baseline_candidate, Mapping):
            return dict(baseline_candidate)
        return {}

    def _translate_guardrail_decision(
        self,
        decision: ai_core_api.GuardrailDecision,
        normalized: NormalizedDocumentPayload,
    ) -> GraphTransition:
        diagnostics: Dict[str, Any] = _json_safe_mapping(decision.attributes)
        policy_events = self._normalize_policy_events(
            diagnostics.get("policy_events")
        ) or tuple(decision.policy_events)
        if policy_events:
            diagnostics["policy_events"] = policy_events
        severity = diagnostics.get("severity")
        if not severity:
            diagnostics["severity"] = "info" if decision.allowed else "error"
        diagnostics.setdefault("guardrail_decision", decision.decision)
        diagnostics.setdefault("tenant_id", normalized.tenant_id)
        diagnostics.setdefault("document_id", normalized.document_id)
        diagnostics["allowed"] = decision.allowed

        if not decision.allowed:
            return _transition("skip_guardrail", decision.reason, diagnostics=diagnostics)
        return _transition("guardrail_allow", decision.reason, diagnostics=diagnostics)

    def _translate_delta_decision(
        self,
        decision: ai_core_api.DeltaDecision,
        normalized: NormalizedDocumentPayload,
        guardrail_transition: Mapping[str, Any] | None,
    ) -> GraphTransition:
        diagnostics: Dict[str, Any] = _json_safe_mapping(decision.attributes)
        policy_events = self._normalize_policy_events(
            diagnostics.get("policy_events")
        )
        diagnostics["policy_events"] = policy_events
        diagnostics.setdefault("severity", "info")
        diagnostics.setdefault("delta_decision", decision.decision)
        if decision.version is not None:
            diagnostics.setdefault("version", decision.version)
        diagnostics.setdefault("tenant_id", normalized.tenant_id)
        diagnostics.setdefault("document_id", normalized.document_id)

        guardrail_events: tuple[str, ...] = ()
        if guardrail_transition is not None:
            diagnostics["guardrail"] = guardrail_transition
            guardrail_diag = guardrail_transition.get("diagnostics", {})
            guardrail_events = self._normalize_policy_events(
                guardrail_diag.get("policy_events")
            )
            diagnostics["policy_events"] = self._merge_policy_events(
                guardrail_events, diagnostics.get("policy_events", ())
            )

        status = decision.decision.strip().lower()
        reason = decision.reason or f"delta_{status or 'unknown'}"
        if status in {"unchanged", "near_duplicate"}:
            if guardrail_events:
                diagnostics["policy_events"] = self._merge_policy_events(
                    guardrail_events, diagnostics.get("policy_events", ())
                )
            return _transition("skip_delta", reason, diagnostics=diagnostics)

        return _transition(
            "upsert",
            reason if reason else "delta_changed",
            diagnostics=diagnostics,
        )

    @staticmethod
    def _normalize_policy_events(value: Any) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return tuple(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, (list, set)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        if value is None:
            return ()
        event = str(value).strip()
        return (event,) if event else ()

    @staticmethod
    def _merge_policy_events(
        *sources: Iterable[str],
    ) -> tuple[str, ...]:
        seen: Dict[str, None] = {}
        for source in sources:
            for event in source:
                key = str(event).strip()
                if not key:
                    continue
                if key not in seen:
                    seen[key] = None
        return tuple(seen.keys())

    # Node: persist_document -------------------------------------------------

    @observe_span(name="upload.ingestion.persist_document", auto_annotate=True)
    def _node_persist_document(self, state: MutableMapping[str, Any]) -> GraphTransition:
        payload = {
            "tenant_id": state["meta"].get("tenant_id"),
            "visibility": state["meta"].get("visibility"),
            "content_hash": state["doc"].get("content_hash"),
            "normalized_text": state["doc"].get("normalized_text"),
        }
        if self._persist is not None:
            result = self._persist(payload)
        else:
            result = {
                "document_id": str(uuid4()),
                "version": state["doc"].get("version") or "1",
            }

        document_id = result.get("document_id")
        version = result.get("version")
        if document_id is None:
            raise UploadIngestionError("persistence_missing_document_id")

        state["doc"].update({"document_id": document_id, "version": version})
        dedupe_key = (
            state["meta"].get("tenant_id"),
            state["meta"].get("source") or "upload",
            state["doc"].get("content_hash"),
        )
        self._dedupe_index[dedupe_key] = {
            "document_id": document_id,
            "version": version,
        }
        return _transition(
            "persist_complete",
            "document_persisted",
            diagnostics={"document_id": document_id, "version": version},
        )

    # Node: chunk_and_embed --------------------------------------------------

    @observe_span(name="upload.ingestion.chunk_and_embed", auto_annotate=True)
    def _node_chunk_and_embed(self, state: MutableMapping[str, Any]) -> GraphTransition:
        normalized_text = state["doc"].get("normalized_text", "")
        words = normalized_text.split()
        chunk = " ".join(words[:128]) if words else normalized_text
        chunks = [chunk] if chunk else []
        state["doc"]["chunks"] = chunks

        embedding_result: Mapping[str, Any]
        if self._embed is not None:
            embedding_result = self._embed(
                {
                    "tenant_id": state["meta"].get("tenant_id"),
                    "document_id": state["doc"].get("document_id"),
                    "chunks": chunks,
                }
            )
        else:
            embedding_result = {"status": "vector_complete", "count": len(chunks)}

        state["doc"]["embedding"] = embedding_result
        return _transition(
            "vector_complete",
            "embedding_triggered",
            diagnostics={"chunks": len(chunks)},
        )

    # Node: lifecycle_hook ---------------------------------------------------

    @observe_span(name="upload.ingestion.lifecycle_hook", auto_annotate=True)
    def _node_lifecycle_hook(self, state: MutableMapping[str, Any]) -> GraphTransition:
        if self._lifecycle_hook is None:
            return _transition("lifecycle_complete", "hook_skipped")

        hook_result = self._lifecycle_hook(
            {
                "tenant_id": state["meta"].get("tenant_id"),
                "document_id": state["doc"].get("document_id"),
                "embedding": state["doc"].get("embedding"),
            }
        )
        state["doc"]["lifecycle"] = hook_result
        return _transition("lifecycle_complete", "hook_completed")

    # Node: finalize ---------------------------------------------------------

    @observe_span(name="upload.ingestion.finalize", auto_annotate=True)
    def _node_finalize(self, state: MutableMapping[str, Any]) -> GraphTransition:
        decision = state["doc"].get("decision")
        if decision and decision.startswith("skip"):
            return _transition("skipped", state["doc"].get("reason", "skipped"))
        return _transition("completed", "ingestion_finished")

    # ---------------------------------------------------------------- Helper

    def _initial_state(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        return {
            "input": dict(payload),
            "meta": {
                "prompt_version": getattr(settings, "PROMPT_VERSION", "v1"),
            },
            "ingest": {},
            "doc": {},
            "telemetry": {"started_at": started, "nodes": {}},
        }

    @staticmethod
    def _compute_total_ms(telemetry: Mapping[str, Any]) -> float:
        total = 0.0
        for entry in telemetry.get("nodes", {}).values():
            took = entry.get("took_ms")
            if took in (None, ""):
                continue
            try:
                total += float(took)
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _span_name(phase: str) -> str:
        return f"upload.ingestion.{phase}"

    def _annotate_span(
        self,
        state: MutableMapping[str, Any],
        *,
        phase: str,
        transition: GraphTransition | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        metadata: Dict[str, Any] = {
            "phase": self._span_name(phase),
        }
        meta_state = state.get("meta", {})
        doc_state = state.get("doc", {})

        tenant_id = self._normalize_optional_str(meta_state.get("tenant_id"))
        if tenant_id:
            metadata["tenant_id"] = tenant_id
        case_id = self._normalize_optional_str(meta_state.get("case_id"))
        if case_id:
            metadata["case_id"] = case_id
        workflow_id = self._normalize_optional_str(meta_state.get("workflow_id"))
        if workflow_id:
            metadata["workflow_id"] = workflow_id
        document_id = doc_state.get("document_id")
        if document_id:
            metadata["document_id"] = str(document_id)
        decision = doc_state.get("decision")
        if decision:
            metadata.setdefault("decision", str(decision))
        if transition is not None:
            metadata["decision"] = transition.decision
            metadata["reason"] = transition.reason
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                metadata[key] = value
        if metadata:
            update_observation(metadata=metadata)

    def _with_transition_metadata(
        self, transition: GraphTransition, state: MutableMapping[str, Any]
    ) -> GraphTransition:
        metadata = self._transition_metadata(state)
        if not metadata:
            return transition
        diagnostics: Dict[str, Any] = dict(transition.diagnostics)
        diagnostics.update(metadata)
        return GraphTransition(
            decision=transition.decision,
            reason=transition.reason,
            diagnostics=diagnostics,
        )

    def _transition_metadata(
        self, state: MutableMapping[str, Any]
    ) -> Dict[str, str]:
        meta_state = state.get("meta", {})
        doc_state = state.get("doc", {})
        metadata: Dict[str, str] = {}

        trace_id = self._normalize_optional_str(meta_state.get("trace_id"))
        if trace_id:
            metadata["trace_id"] = trace_id
        workflow_id = self._normalize_optional_str(meta_state.get("workflow_id"))
        if workflow_id:
            metadata["workflow_id"] = workflow_id
        document_id = doc_state.get("document_id")
        if document_id:
            metadata["document_id"] = str(document_id)
        return metadata

    def _maybe_emit_transition_event(
        self,
        phase: str,
        transition: GraphTransition,
        state: MutableMapping[str, Any],
    ) -> None:
        diagnostics = transition.diagnostics
        severity = str(diagnostics.get("severity") or "").strip().lower()
        decision_text = transition.decision.strip().lower()
        if severity not in {"error"} and "deny" not in decision_text:
            return

        payload: Dict[str, Any] = {
            "event": "upload.ingestion.denied"
            if "deny" in decision_text
            else "upload.ingestion.error",
            "phase": self._span_name(phase),
            "decision": transition.decision,
            "reason": transition.reason,
        }
        payload.update(self._collect_observability_metadata(state))
        if diagnostics:
            payload["diagnostics"] = _make_json_safe(diagnostics)
        emit_event(payload)

    def _handle_node_error(
        self,
        phase: str,
        exc: Exception,
        state: MutableMapping[str, Any],
    ) -> None:
        error_payload = {
            "event": "upload.ingestion.error",
            "phase": self._span_name(phase),
            "error": exc.__class__.__name__,
            "message": str(exc),
        }
        error_payload.update(self._collect_observability_metadata(state))
        emit_event(error_payload)
        self._annotate_span(
            state,
            phase=phase,
            extra={
                "error": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )

    def _collect_observability_metadata(
        self, state: MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        meta_state = state.get("meta", {})
        doc_state = state.get("doc", {})

        tenant_id = self._normalize_optional_str(meta_state.get("tenant_id"))
        if tenant_id:
            metadata["tenant_id"] = tenant_id
        case_id = self._normalize_optional_str(meta_state.get("case_id"))
        if case_id:
            metadata["case_id"] = case_id
        trace_id = self._normalize_optional_str(meta_state.get("trace_id"))
        if trace_id:
            metadata["trace_id"] = trace_id
        workflow_id = self._normalize_optional_str(meta_state.get("workflow_id"))
        if workflow_id:
            metadata["workflow_id"] = workflow_id
        document_id = doc_state.get("document_id")
        if document_id:
            metadata["document_id"] = str(document_id)
        version = doc_state.get("version")
        if version:
            metadata["version"] = str(version)
        decision = doc_state.get("decision")
        if decision:
            metadata["decision"] = str(decision)
        return metadata

    @staticmethod
    def _require_str(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise UploadIngestionError(f"input_missing:{key}")
        return value.strip()

    @staticmethod
    def _normalize_mime(value: object | None) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _normalize_tags(value: object | None) -> Iterable[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, Iterable):
            tags: list[str] = []
            for entry in value:
                if entry is None:
                    continue
                tag = str(entry).strip()
                if tag:
                    tags.append(tag)
            return tags
        return []

    @staticmethod
    def _normalize_optional_str(value: object | None) -> str | None:
        if value is None:
            return None
        candidate = str(value).strip()
        return candidate or None

    @staticmethod
    def _resolve_visibility(value: object | None) -> str:
        if value is None:
            return "active"
        candidate = str(value).strip().lower()
        return candidate or "active"


__all__ = ["UploadIngestionGraph", "UploadIngestionError", "GraphTransition"]

