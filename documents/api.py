"""High level document service helpers consumed by orchestration graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from common.object_store import ObjectStore, get_default_object_store
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
    NormalizedDocumentInputV1,
)
from documents.repository import (
    DocumentLifecycleRecord,
    DocumentLifecycleStore,
    DEFAULT_LIFECYCLE_STORE,
)
from documents.normalization import normalized_primary_text


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
    return get_default_object_store()


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
    content_raw: str = field(default="", repr=False)
    content_normalized: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        raw_value = self.content_raw or ""
        normalized_value = self.content_normalized or self.primary_text or ""
        normalized_value = normalized_value.strip()
        object.__setattr__(self, "content_raw", raw_value)
        object.__setattr__(self, "content_normalized", normalized_value)
        primary_text = (self.primary_text or "").strip()
        if normalized_value and primary_text != normalized_value:
            object.__setattr__(self, "primary_text", normalized_value)
        elif not primary_text and normalized_value:
            object.__setattr__(self, "primary_text", normalized_value)

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
        return {
            "document": self.document.model_dump(),
            "primary_text": self.primary_text,
            "checksum": self.checksum,
            "metadata": dict(self.metadata),
            "content_raw": self.content_raw,
            "content_normalized": self.content_normalized,
        }


@dataclass(frozen=True)
class LifecycleStatusUpdate:
    """Lifecycle mutation returned by :func:`update_lifecycle_status`."""

    record: DocumentLifecycleRecord

    @property
    def status(self) -> str:
        return self.record.state

    def to_dict(self) -> Mapping[str, Any]:
        return dict(self.record.as_payload())


def normalize_from_raw(
    *,
    contract: NormalizedDocumentInputV1,
    object_store: ObjectStore | None = None,
) -> NormalizedDocumentPayload:
    """Normalize crawler raw payloads into a :class:`NormalizedDocument`."""

    if not isinstance(contract, NormalizedDocumentInputV1):
        raise TypeError("normalized_document_input_required")

    store = _resolve_object_store(object_store)

    def _read_from_store(path: str) -> bytes:
        normalized_path = _validate_object_store_path(path, store)
        try:
            return store.read_bytes(normalized_path)
        except FileNotFoundError as exc:  # pragma: no cover - defensive guard
            raise ValueError("raw_content_missing") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError("payload_path_invalid") from exc

    object_reader = _read_from_store if contract.requires_object_store else None
    try:
        payload_bytes = contract.resolve_payload_bytes(
            object_store_reader=object_reader
        )
    except ValueError as exc:
        if str(exc) == "object_store_reader_required":
            raise ValueError("payload_path_invalid") from exc
        raise

    content = contract.resolve_payload_text(payload_bytes=payload_bytes)
    checksum = contract.compute_checksum(payload_bytes)
    payload_base64 = contract.payload_base64(payload_bytes)

    document_id = _coerce_uuid(contract.document_id)
    document_ref = DocumentRef(
        tenant_id=contract.tenant_id,
        workflow_id=contract.workflow_id,
        document_id=document_id,
        collection_id=contract.collection_id,
        version=contract.version,
    )

    parse_stats: dict[str, Any] = {
        "parser.character_count": len(content),
        "parser.token_count": len(content.split()),
        "crawler.primary_text_hash_sha256": checksum,
    }

    doc_meta = DocumentMeta(
        tenant_id=contract.tenant_id,
        workflow_id=contract.workflow_id,
        title=contract.title,
        language=contract.language,
        tags=contract.tags,
        document_collection_id=contract.document_collection_id,
        origin_uri=contract.origin_uri,
        external_ref=contract.build_external_reference(document_id),
        parse_stats=parse_stats,
    )

    payload_size = contract.payload_size(payload_bytes)
    blob = InlineBlob(
        type="inline",
        media_type=contract.media_type,
        base64=payload_base64,
        sha256=checksum,
        size=payload_size,
    )

    document = NormalizedDocument(
        ref=document_ref,
        meta=doc_meta,
        blob=blob,
        checksum=checksum,
        created_at=_timestamp(),
        source=contract.source,
        lifecycle_state="active",
        assets=[],
    )

    normalized_text = normalized_primary_text(content)

    metadata: dict[str, Any] = dict(contract.metadata_payload)
    metadata.update(
        {
            "document_id": str(document_id),
            "collection_id": (
                str(contract.collection_id)
                if contract.collection_id is not None
                else None
            ),
            "document_collection_id": (
                str(contract.document_collection_id)
                if contract.document_collection_id is not None
                else None
            ),
            "external_id": contract.external_id,
            "version": contract.version,
            "media_type": contract.media_type,
            "payload_size": payload_size,
        }
    )

    if contract.tags:
        metadata["tags"] = list(contract.tags)

    metadata = {k: v for k, v in metadata.items() if v is not None}

    return NormalizedDocumentPayload(
        document=document,
        primary_text=normalized_text,
        payload_bytes=payload_bytes,
        metadata=dict(metadata),
        content_raw=content,
        content_normalized=normalized_text,
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
    "NormalizedDocumentInputV1",
    "LifecycleStatusUpdate",
    "NormalizedDocumentPayload",
    "set_object_store",
    "normalize_from_raw",
    "update_lifecycle_status",
]
