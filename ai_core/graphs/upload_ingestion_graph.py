"""Upload ingestion graph orchestrating document processing for uploads."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping
from uuid import uuid4

from django.conf import settings


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
        guardrail_enforcer: Callable[[Mapping[str, Any]], GraphTransition] | None = None,
        delta_decider: Callable[[Mapping[str, Any]], GraphTransition] | None = None,
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

    def run(self, payload: Mapping[str, Any], run_until: str | None = None) -> Mapping[str, Any]:
        """Execute the upload ingestion graph for *payload*."""

        state = self._initial_state(payload)
        target_node = self._RUN_UNTIL_TO_NODE.get(run_until or "")
        telemetry = state["telemetry"]
        transitions: dict[str, Mapping[str, Any]] = {}

        for node_name in self._RUN_ORDER:
            started = datetime.now(timezone.utc)
            transition = self._execute_node(node_name, state)
            ended = datetime.now(timezone.utc)
            elapsed_ms = (ended - started).total_seconds() * 1000
            telemetry.setdefault("nodes", {})[node_name] = {
                "took_ms": elapsed_ms,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
            }
            transitions[node_name] = transition.to_dict()
            state["doc"]["last_transition"] = transition.to_dict()
            state["doc"]["decision"] = transition.decision
            state["doc"]["reason"] = transition.reason

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
        return result

    # -------------------------------------------------------------- Node impl

    def _execute_node(self, name: str, state: MutableMapping[str, Any]) -> GraphTransition:
        try:
            handler = getattr(self, f"_node_{name}")
        except AttributeError as exc:  # pragma: no cover - defensive
            raise UploadIngestionError(f"node_missing:{name}") from exc
        try:
            return handler(state)
        except UploadIngestionError:
            raise
        except Exception as exc:
            raise UploadIngestionError(f"node_failed:{name}") from exc

    # Node: accept_upload ----------------------------------------------------

    def _node_accept_upload(self, state: MutableMapping[str, Any]) -> GraphTransition:
        payload = state["input"]
        tenant_id = self._require_str(payload, "tenant_id")
        uploader_id = self._require_str(payload, "uploader_id")
        visibility = self._resolve_visibility(payload.get("visibility"))

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
                "tags": tuple(self._normalize_tags(payload.get("tags"))),
                "source_key": self._normalize_optional_str(payload.get("source_key")),
                "filename": filename,
            }
        )
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

    def _node_deduplicate(self, state: MutableMapping[str, Any]) -> GraphTransition:
        binary: bytes = state["ingest"].get("binary", b"")
        tenant_id = state["meta"].get("tenant_id")
        visibility = state["meta"].get("visibility")
        content_hash = hashlib.sha256(binary).hexdigest()
        state["ingest"]["content_hash"] = content_hash

        dedupe_key = (tenant_id, visibility, content_hash)
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
            version_key = (tenant_id, visibility, source_key)
            version = self._source_versions.get(version_key, 0) + 1
            self._source_versions[version_key] = version
            state["doc"]["version"] = str(version)

        state["doc"]["content_hash"] = content_hash
        return _transition("proceed", "dedupe_ok", diagnostics={"content_hash": content_hash})

    # Node: parse ------------------------------------------------------------

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

    def _node_normalize(self, state: MutableMapping[str, Any]) -> GraphTransition:
        raw_text = state["doc"].get("raw_text", "")
        normalized = " ".join(raw_text.split()) if raw_text else ""
        state["doc"]["normalized_text"] = normalized
        return _transition("normalize_complete", "normalized", diagnostics={"length": len(normalized)})

    # Node: delta and guardrails --------------------------------------------

    def _node_delta_and_guardrails(self, state: MutableMapping[str, Any]) -> GraphTransition:
        payload = {
            "tenant_id": state["meta"].get("tenant_id"),
            "content_hash": state["doc"].get("content_hash"),
            "normalized_text": state["doc"].get("normalized_text"),
            "version": state["doc"].get("version"),
        }
        if self._guardrail_enforcer is not None:
            guardrail = self._guardrail_enforcer(payload)
            if not isinstance(guardrail, GraphTransition):
                raise UploadIngestionError("guardrail_transition_invalid")
            if not guardrail.decision.startswith("allow"):
                return guardrail

        if self._delta_decider is not None:
            delta = self._delta_decider(payload)
            if not isinstance(delta, GraphTransition):
                raise UploadIngestionError("delta_transition_invalid")
            state["doc"]["delta"] = delta.to_dict()
            if delta.decision == "skip_guardrail":
                return delta

        return _transition("upsert", "delta_ok")

    # Node: persist_document -------------------------------------------------

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
            state["meta"].get("visibility"),
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
            total += float(entry.get("took_ms", 0.0))
        return total

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

