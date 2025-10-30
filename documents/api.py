"""High level document service helpers consumed by orchestration graphs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Optional
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from common.object_store import ObjectStore, get_default_object_store
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.repository import (
    DocumentLifecycleRecord,
    DocumentLifecycleStore,
    DEFAULT_LIFECYCLE_STORE,
)


_OBJECT_STORE_OVERRIDE: ObjectStore | None = None


def set_object_store(store: ObjectStore | None) -> None:
    """Override the object store used by :mod:`documents.api`."""

    global _OBJECT_STORE_OVERRIDE
    _OBJECT_STORE_OVERRIDE = store


def _resolve_object_store(store: ObjectStore | None = None) -> ObjectStore:
    if store is not None:
        return store
    if _OBJECT_STORE_OVERRIDE is not None:
        return _OBJECT_STORE_OVERRIDE
    try:
        return get_default_object_store()
    except RuntimeError:
        importlib.import_module("ai_core.infra.object_store")
        return get_default_object_store()



def _normalize_mapping(
    value: Mapping[str, Any] | MutableMapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        if isinstance(value, MutableMapping):
            value = dict(value)
        else:
            raise TypeError("metadata_mapping_required")
    return MappingProxyType(dict(value))


def _normalize_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("raw_content_decode_error") from exc
    raise TypeError("raw_content_type")


def _coerce_payload_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return b""
        try:
            return base64.b64decode(candidate, validate=True)
        except (binascii.Error, ValueError):
            return candidate.encode("utf-8")
    raise TypeError("payload_bytes_type")


def _extract_charset(value: Any) -> Optional[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        for param in candidate.split(";")[1:]:
            key, _, param_value = param.partition("=")
            if key.strip().lower() == "charset":
                encoding = param_value.strip().strip('"').strip("'")
                if encoding:
                    return encoding
    return None


def _decode_payload_text(payload: bytes, encoding_hint: Optional[str]) -> str:
    if encoding_hint:
        try:
            return payload.decode(encoding_hint)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("utf-8", errors="replace")


def _validate_object_store_path(value: str, store: ObjectStore) -> str:
    """Return a safe object store path relative to the configured object store."""

    candidate = Path(value)
    if candidate.is_absolute() or candidate.anchor:
        raise ValueError("payload_path_invalid")

    base_path = store.BASE_PATH.resolve()
    try:
        resolved = (base_path / candidate).resolve()
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("payload_path_invalid") from exc

    if not resolved.is_relative_to(base_path):
        raise ValueError("payload_path_invalid")

    try:
        relative_path = resolved.relative_to(base_path)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("payload_path_invalid") from exc

    if str(relative_path) == ".":
        raise ValueError("payload_path_invalid")

    return relative_path.as_posix()


def _resolve_payload(
    raw_reference: Mapping[str, Any], store: ObjectStore
) -> tuple[bytes, str]:
    if "content" in raw_reference:
        content = _normalize_text(raw_reference.get("content"))
        return content.encode("utf-8"), content

    payload_bytes: Optional[bytes] = None
    if "payload_path" in raw_reference:
        payload_path = raw_reference.get("payload_path")
        if isinstance(payload_path, (str, PathLike)):
            candidate = str(payload_path).strip()
            if candidate:
                try:
                    normalized_path = _validate_object_store_path(candidate, store)
                except ValueError as exc:
                    raise ValueError("payload_path_invalid") from exc
                try:
                    payload_bytes = store.read_bytes(normalized_path)
                except FileNotFoundError as exc:  # pragma: no cover - defensive guard
                    raise ValueError("raw_content_missing") from exc
                except Exception as exc:  # pragma: no cover - defensive guard
                    raise ValueError("payload_path_invalid") from exc
        elif payload_path is not None:
            raise TypeError("payload_path_type")

    if payload_bytes is None and "payload_bytes" in raw_reference:
        payload_bytes = _coerce_payload_bytes(raw_reference.get("payload_bytes"))
    elif payload_bytes is None and "payload_base64" in raw_reference:
        base64_value = raw_reference.get("payload_base64")
        if isinstance(base64_value, (str, bytes, bytearray)):
            payload_bytes = _coerce_payload_bytes(base64_value)

    if payload_bytes is None:
        raise ValueError("raw_content_missing")

    encoding_hint = raw_reference.get("payload_encoding") or raw_reference.get(
        "content_encoding"
    )
    if encoding_hint is None:
        metadata = raw_reference.get("metadata")
        metadata_mapping: Optional[Mapping[str, Any]] = (
            metadata if isinstance(metadata, Mapping) else None
        )

        for candidate in (
            raw_reference.get("content_type"),
            raw_reference.get("media_type"),
            raw_reference.get("mime_type"),
            metadata_mapping.get("content_type") if metadata_mapping else None,
            metadata_mapping.get("content-type") if metadata_mapping else None,
        ):
            encoding_hint = _extract_charset(candidate)
            if encoding_hint:
                break
    content = _decode_payload_text(payload_bytes, encoding_hint)
    return payload_bytes, content


def _coerce_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    if value is None:
        return uuid4()
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return uuid5(NAMESPACE_URL, str(value))


def _timestamp() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class NormalizedDocumentPayload:
    """Wrapper exposing the normalized document and derived artefacts."""

    document: NormalizedDocument
    primary_text: str
    payload_bytes: bytes
    metadata: Mapping[str, Any]

    @property
    def tenant_id(self) -> str:
        return self.document.ref.tenant_id

    @property
    def document_id(self) -> str:
        return str(self.document.ref.document_id)

    @property
    def checksum(self) -> str:
        return self.document.checksum

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "document": self.document.model_dump(),
                "primary_text": self.primary_text,
                "checksum": self.checksum,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class LifecycleStatusUpdate:
    """Lifecycle mutation returned by :func:`update_lifecycle_status`."""

    record: DocumentLifecycleRecord

    @property
    def status(self) -> str:
        return self.record.state

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(self.record.as_payload())


def normalize_from_raw(
    *,
    raw_reference: Mapping[str, Any],
    tenant_id: str,
    case_id: Optional[str] = None,
    request_id: Optional[str] = None,
    object_store: ObjectStore | None = None,
) -> NormalizedDocumentPayload:
    """Normalize crawler raw payloads into a :class:`NormalizedDocument`."""

    if not isinstance(raw_reference, Mapping):
        raise TypeError("raw_reference_mapping_required")

    store = _resolve_object_store(object_store)
    payload_bytes, content = _resolve_payload(raw_reference, store)
    media_type = (
        str(
            raw_reference.get("media_type")
            or raw_reference.get("content_type")
            or raw_reference.get("mime_type")
            or raw_reference.get("metadata", {}).get("media_type")
            or "text/plain"
        )
        .strip()
        .lower()
    )

    checksum = hashlib.sha256(payload_bytes).hexdigest()
    payload_base64 = base64.b64encode(payload_bytes).decode("ascii")

    metadata = _normalize_mapping(raw_reference.get("metadata"))
    workflow_id = str(
        metadata.get("workflow_id") or raw_reference.get("workflow_id") or "crawler"
    )
    workflow_id = workflow_id.strip() or "crawler"

    document_id = _coerce_uuid(raw_reference.get("document_id"))
    collection_id = metadata.get("collection_id")
    if collection_id is not None:
        try:
            collection_id = UUID(str(collection_id))
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            collection_id = None

    version = metadata.get("version")
    version_str = str(version).strip() if version else None

    document_ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        collection_id=collection_id,
        version=version_str,
    )

    external_ref: dict[str, str] = {}
    provider = str(
        metadata.get("provider") or raw_reference.get("provider") or "crawler"
    ).strip()
    if provider:
        external_ref["provider"] = provider
    external_id = metadata.get("external_id") or raw_reference.get("external_id")
    if external_id:
        external_ref["external_id"] = str(external_id).strip()
    else:
        external_ref["external_id"] = f"{provider}:{document_id}"
    if case_id:
        external_ref["case_id"] = str(case_id)

    origin_uri = metadata.get("origin_uri") or raw_reference.get("origin_uri")

    parse_stats: dict[str, Any] = {
        "parser.character_count": len(content),
        "parser.token_count": len(content.split()),
        "crawler.primary_text_hash_sha256": checksum,
    }

    tags = metadata.get("tags") or ()
    if isinstance(tags, (list, tuple, set)):
        normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    else:
        normalized_tags = []

    doc_meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        title=metadata.get("title"),
        language=metadata.get("language"),
        tags=normalized_tags,
        origin_uri=str(origin_uri).strip() or None,
        external_ref=external_ref,
        parse_stats=parse_stats,
    )

    blob = InlineBlob(
        type="inline",
        media_type=media_type or "text/plain",
        base64=payload_base64,
        sha256=checksum,
        size=len(payload_bytes),
    )

    document = NormalizedDocument(
        ref=document_ref,
        meta=doc_meta,
        blob=blob,
        checksum=checksum,
        created_at=_timestamp(),
        source=str(raw_reference.get("source") or "crawler"),
        lifecycle_state="active",
        assets=[],
    )

    metadata_payload: dict[str, Any] = {
        "tenant_id": tenant_id,
        "workflow_id": workflow_id,
        "case_id": case_id,
        "request_id": request_id,
        "provider": provider or None,
        "origin_uri": doc_meta.origin_uri,
    }

    return NormalizedDocumentPayload(
        document=document,
        primary_text=content.strip(),
        payload_bytes=payload_bytes,
        metadata=MappingProxyType(
            {k: v for k, v in metadata_payload.items() if v is not None}
        ),
    )


def update_lifecycle_status(
    *,
    tenant_id: str,
    document_id: str | UUID,
    status: str,
    previous_status: Optional[str] = None,
    workflow_id: Optional[str] = None,
    reason: Optional[str] = None,
    policy_events: Optional[Mapping[str, Any] | tuple[str, ...] | list[str]] = None,
    store: DocumentLifecycleStore = DEFAULT_LIFECYCLE_STORE,
) -> LifecycleStatusUpdate:
    """Persist a lifecycle transition in the shared lifecycle store."""

    if not status:
        raise ValueError("status_required")

    document_uuid = _coerce_uuid(document_id)

    events: tuple[str, ...]
    if policy_events is None:
        events = ()
    elif isinstance(policy_events, tuple):
        events = policy_events
    elif isinstance(policy_events, list):
        events = tuple(str(item) for item in policy_events if str(item).strip())
    elif isinstance(policy_events, Mapping):
        events = tuple(str(key) for key in policy_events.keys() if str(key).strip())
    else:
        events = (str(policy_events),)

    record = store.record_document_state(
        tenant_id=tenant_id,
        document_id=document_uuid,
        workflow_id=str(workflow_id).strip() or None,
        state=str(status).strip(),
        reason=reason or status,
        policy_events=events,
        changed_at=_timestamp(),
    )

    return LifecycleStatusUpdate(record)


__all__ = [
    "LifecycleStatusUpdate",
    "NormalizedDocumentPayload",
    "set_object_store",
    "normalize_from_raw",
    "update_lifecycle_status",
]
