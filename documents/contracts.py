"""Contracts for normalized documents and assets.

Terminology used across ingestion flows:

- **provider** – Canonical name of the upstream system that supplied or
  generated the document (e.g. ``web``, ``servicenow``). The value is persisted
  inside :attr:`DocumentMeta.external_ref` under the ``provider`` key.
- **source** – Channel that handed the document to the NOESIS ingestion stack.
  Valid values describe the transport such as ``crawler`` or ``upload`` and are
  stored on :class:`NormalizedDocument.source`.
- **process** – Business process alias that steers routing and embedding
  profiles. It is propagated alongside ingestion metadata (e.g. chunk
  envelopes) and defaults to the ``source`` when callers do not provide an
  explicit value.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from datetime import datetime, timezone
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .contract_utils import (
    is_bcp47_like,
    normalize_media_type,
    normalize_optional_string,
    normalize_string,
    normalize_tags,
    normalize_tenant,
    normalize_workflow_id,
    normalize_title,
    truncate_text,
    validate_bbox,
    resolve_workflow_id,
)

from .contracts_context import (
    asset_media_guard,
    get_asset_media_guard,
    is_strict_checksums_enabled,
    set_asset_media_guard,
    set_strict_checksums,
    strict_checksums,
)


__all__ = [
    "asset_media_guard",
    "get_asset_media_guard",
    "is_strict_checksums_enabled",
    "set_asset_media_guard",
    "set_strict_checksums",
    "strict_checksums",
    "BlobLocator",
    "ExternalBlob",
    "FileBlob",
    "InlineBlob",
    "LocalFileBlob",
]


_HEX_64_RE = re.compile(r"^[a-f0-9]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_CAPTION_SOURCES = {
    "none",
    "alt_text",
    "figure_caption",
    "context_before",
    "context_after",
    "notes",
    "origin",
    "vlm",
    "ocr",
    "manual",
}

CaptionSource = Literal[
    "none",
    "alt_text",
    "figure_caption",
    "context_before",
    "context_after",
    "notes",
    "origin",
    "vlm",
    "ocr",
    "manual",
]


# Default provider names resolved from ingestion sources. Callers may override
# these values via metadata overrides passed to the ingestion interfaces.
DEFAULT_PROVIDER_BY_SOURCE = {
    "crawler": "web",
    "upload": "upload",
    "integration": "integration",
    "other": "other",
}


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover - propagate message
        raise ValueError("base64_invalid") from exc


def _extract_charset_param(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    for param in candidate.split(";")[1:]:
        key, _, param_value = param.partition("=")
        if key.strip().lower() == "charset":
            encoding = param_value.strip().strip('"').strip("'")
            if encoding:
                return encoding
    return None


def _coerce_uuid(value: Any):
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        candidate = normalize_string(value)
        if not candidate:
            raise ValueError("uuid_empty")
        try:
            return UUID(candidate)
        except ValueError as exc:  # pragma: no cover - propagate code
            raise ValueError("uuid_invalid") from exc
    raise TypeError("uuid_type")


class DocumentRef(BaseModel):
    """Stable reference to a stored document."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        title="Document Reference",
        description="Identifies a document within a tenant and optional collection.",
    )

    tenant_id: str = Field(description="Tenant identifier owning the document.")
    workflow_id: str = Field(
        description="Workflow identifier that produced or manages the document."
    )
    document_id: UUID = Field(description="Unique identifier of the document.")
    collection_id: Optional[UUID] = Field(
        default=None, description="Optional collection containing the document."
    )
    document_collection_id: Optional[UUID] = Field(
        default=None,
        description="Optional logical DocumentCollection identifier for the document.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Optional semantic version marker for the ingested document.",
        max_length=64,
    )

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalize_tenant_id(cls, value: str) -> str:
        return normalize_tenant(value)

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _normalize_workflow_id(cls, value: str) -> str:
        return normalize_workflow_id(value)

    @field_validator(
        "document_id", "collection_id", "document_collection_id", mode="before"
    )
    @classmethod
    def _coerce_uuid_fields(cls, value):
        return _coerce_uuid(value)

    @field_validator("version", mode="before")
    @classmethod
    def _normalize_version(cls, value: Optional[str]) -> Optional[str]:
        normalized = normalize_optional_string(value)
        if normalized is None:
            return None
        if len(normalized) > 64:
            raise ValueError("version_too_long")
        if not _VERSION_RE.fullmatch(normalized):
            raise ValueError("version_invalid")
        return normalized


class FileBlob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["file"] = Field(description="Discriminator identifying a file blob.")
    uri: str = Field(description="URI for fetching the blob from storage.")
    sha256: str = Field(
        description="Hex encoded SHA-256 checksum of the blob contents."
    )
    size: int = Field(description="Blob size in bytes.")
    media_type: Optional[str] = Field(
        default=None,
        description="RFC compliant media type of the blob contents (type/subtype lowercase).",
    )

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        normalized = normalize_string(value)
        if not normalized:
            raise ValueError("uri_empty")
        return normalized

    @field_validator("sha256")
    @classmethod
    def _validate_sha(cls, value: str) -> str:
        if not _HEX_64_RE.fullmatch(value):
            raise ValueError("sha256_invalid")
        return value

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("size_negative")
        return value


class LocalFileBlob(BaseModel):
    """Temporary local file reference for processing."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["local_file"] = Field(
        description="Discriminator identifying a local file blob."
    )
    path: str = Field(description="Absolute path to the local file.")
    media_type: Optional[str] = Field(
        default=None, description="Optional media type hint."
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        s = normalize_string(value)
        if not s:
            raise ValueError("path_empty")
        return s


class ExternalBlob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["external"] = Field(
        description="Discriminator identifying an external blob reference."
    )
    kind: Literal["s3", "gcs", "http", "https"] = Field(
        description="Type of external storage backend for the blob (http/https for web URLs)."
    )
    uri: str = Field(description="External URI or object identifier.")
    sha256: Optional[str] = Field(
        default=None,
        description="Optional checksum when provided by the external source.",
    )

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        normalized = normalize_string(value)
        if not normalized:
            raise ValueError("uri_empty")
        return normalized

    @field_validator("sha256")
    @classmethod
    def _validate_sha(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not _HEX_64_RE.fullmatch(value):
            raise ValueError("sha256_invalid")
        return value


class InlineBlob(BaseModel):
    """Inline blob with base64-encoded payload for small uploads."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["inline"] = Field(
        description="Discriminator identifying an inline blob."
    )
    media_type: str = Field(description="MIME type of the blob payload.")
    base64: str = Field(description="Base64-encoded blob content.")
    sha256: str = Field(
        description="Hex encoded SHA-256 checksum of the decoded blob contents."
    )
    size: int = Field(description="Size of the decoded blob in bytes.")

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str) -> str:
        normalized = normalize_media_type(value)
        if not normalized:
            raise ValueError("media_type_empty")
        return normalized

    @field_validator("base64")
    @classmethod
    def _validate_base64(cls, value: str) -> str:
        normalized = normalize_string(value)
        if not normalized:
            raise ValueError("base64_empty")
        return normalized

    @field_validator("sha256")
    @classmethod
    def _validate_sha(cls, value: str) -> str:
        if not _HEX_64_RE.fullmatch(value):
            raise ValueError("sha256_invalid")
        return value

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("size_negative")
        return value

    @model_validator(mode="after")
    def _validate_integrity(self) -> "InlineBlob":
        if not is_strict_checksums_enabled():
            return self

        try:
            payload = base64.b64decode(self.base64, validate=True)
        except Exception as exc:
            raise ValueError("base64_invalid") from exc

        if len(payload) != self.size:
            raise ValueError("size_mismatch")

        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != self.sha256:
            raise ValueError("checksum_mismatch")

        return self

    def decoded_payload(self) -> bytes:
        """Decode and return the base64 payload as bytes."""
        return base64.b64decode(self.base64)


BlobLocator = Annotated[
    Union[FileBlob, ExternalBlob, LocalFileBlob, InlineBlob],
    Field(
        discriminator="type",
        title="Blob Locator",
        description="Discriminated union pointing to stored or external blobs.",
    ),
]


class DocumentMeta(BaseModel):
    """Metadata captured during ingestion of a document."""

    model_config = ConfigDict(
        extra="forbid",
        title="Document Metadata",
        description="Describes the document title, language and ingestion context.",
    )

    tenant_id: str = Field(description="Tenant identifier the metadata belongs to.")
    workflow_id: str = Field(
        description="Workflow identifier that groups document ingestion context."
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional human readable document title.",
        max_length=256,
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional BCP-47 language tag representing the primary language.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional collection of tags for retrieval and filtering.",
    )
    document_collection_id: Optional[UUID] = Field(
        default=None,
        description="Optional logical DocumentCollection identifier for the document.",
    )
    origin_uri: Optional[str] = Field(
        default=None,
        description="Original source URI for the ingested document.",
    )
    crawl_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the document was crawled or collected.",
    )
    external_ref: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional external reference identifiers provided by source systems "
            "(<= 16 entries; keys <= 128 chars, values <= 512 chars; limits enforced). "
            "The keys `provider` and `external_id` capture the upstream system and "
            "its native identifier. Additional provider specific attributes must "
            "use the `provider_tag:` prefix."
        ),
    )
    parse_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional parser statistics captured during ingestion (string keyed, "
            "JSON-serialisable values). Recommended namespaces: `parser.*` for "
            "text metrics, `assets.*` for media metrics, and `parse.*` for "
            "framework-provided counters."
        ),
    )
    pipeline_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional runtime overrides for the document pipeline configuration "
            "(string keyed, JSON-serialisable values)."
        ),
    )

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalize_tenant_id(cls, value: str) -> str:
        return normalize_tenant(value)

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _normalize_workflow_id(cls, value: str) -> str:
        return normalize_workflow_id(value)

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title(cls, value: Optional[str]) -> Optional[str]:
        return normalize_title(value)

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = normalize_optional_string(value)
        if normalized is None:
            return None
        if not is_bcp47_like(normalized):
            raise ValueError("language_invalid")
        return normalized

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        raise TypeError("tags_type")

    @field_validator("document_collection_id", mode="before")
    @classmethod
    def _coerce_document_collection_id_meta(
        cls, value: Optional[UUID] | str
    ) -> Optional[UUID]:
        if value is None:
            return None
        if isinstance(value, UUID):
            return value
        candidate = normalize_optional_string(value)
        if not candidate:
            return None
        return UUID(candidate)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: List[str]) -> List[str]:
        return normalize_tags(value)

    @field_validator("origin_uri", mode="before")
    @classmethod
    def _normalize_origin_uri(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("external_ref", mode="before")
    @classmethod
    def _normalize_external_ref(
        cls, value: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        if value is None:
            return None
        if len(value) > 16:
            raise ValueError("external_ref_too_many")
        normalized: Dict[str, str] = {}
        for key, val in value.items():
            norm_key = normalize_string(str(key))
            if not norm_key:
                raise ValueError("external_ref_key_empty")
            if len(norm_key) > 128:
                raise ValueError("external_ref_key_too_long")
            norm_val = normalize_string(str(val))
            if not norm_val:
                raise ValueError("external_ref_value_empty")
            if len(norm_val) > 512:
                raise ValueError("external_ref_value_too_long")
            normalized[norm_key] = norm_val
        return normalized

    @field_validator("crawl_timestamp")
    @classmethod
    def _ensure_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("crawl_timestamp_naive")
        return value.astimezone(timezone.utc)

    @field_validator("parse_stats", mode="before")
    @classmethod
    def _normalize_parse_stats(
        cls, value: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("parse_stats_type")
        normalized: Dict[str, Any] = {}
        for key, raw in value.items():
            key_str = normalize_string(str(key))
            if not key_str:
                raise ValueError("parse_stats_key")
            normalized[key_str] = cls._coerce_parse_stat_value(raw)
        return normalized

    @classmethod
    def _coerce_parse_stat_value(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("parse_stats_value")
            return float(value)
        if isinstance(value, str):
            return normalize_string(value)
        if isinstance(value, (list, tuple)):
            return [cls._coerce_parse_stat_value(item) for item in value]
        if isinstance(value, Mapping):
            nested: Dict[str, Any] = {}
            for key, raw in value.items():
                key_str = normalize_string(str(key))
                if not key_str:
                    raise ValueError("parse_stats_key")
                nested[key_str] = cls._coerce_parse_stat_value(raw)
            return nested
        raise ValueError("parse_stats_value")

    @field_validator("pipeline_config", mode="before")
    @classmethod
    def _normalize_pipeline_config(
        cls, value: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("pipeline_config_type")
        normalized: Dict[str, Any] = {}
        for key, raw in value.items():
            key_str = normalize_string(str(key))
            if not key_str:
                raise ValueError("pipeline_config_key")
            normalized[key_str] = raw
        return normalized


class AssetRef(BaseModel):
    """Reference to an extracted asset belonging to a document."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        title="Asset Reference",
        description="Identifies an asset and its parent document within a tenant.",
    )

    tenant_id: str = Field(description="Tenant identifier owning the asset.")
    workflow_id: str = Field(
        description="Workflow identifier linking the asset to its document workflow."
    )
    asset_id: UUID = Field(description="Unique identifier of the asset.")
    document_id: UUID = Field(description="Identifier of the parent document.")
    collection_id: Optional[UUID] = Field(
        default=None, description="Optional collection identifier for the asset."
    )

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalize_tenant_id(cls, value: str) -> str:
        return normalize_tenant(value)

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _normalize_workflow_id(cls, value: str) -> str:
        return normalize_workflow_id(value)

    @field_validator("asset_id", "document_id", "collection_id", mode="before")
    @classmethod
    def _coerce_uuid_fields(cls, value):
        return _coerce_uuid(value)


class DocumentBlobDescriptorV1(BaseModel):
    """Descriptor describing the payload supplied for ingestion."""

    model_config = ConfigDict(
        extra="forbid",
        title="Document Blob Descriptor (v1)",
        description="Structured payload descriptor supporting inline text, raw bytes or stored objects.",
    )

    _payload_cache: Optional[bytes] = PrivateAttr(default=None)

    inline_text: Optional[str] = Field(
        default=None,
        description="Inline UTF-8 text payload supplied directly by the caller.",
    )
    payload_bytes: Optional[bytes] = Field(
        default=None,
        description="Binary payload supplied as bytes.",
    )
    payload_base64: Optional[str] = Field(
        default=None,
        description="Payload provided as a base64 encoded string.",
    )
    object_store_path: Optional[str] = Field(
        default=None,
        description="Relative object store path pointing to the payload.",
    )
    media_type: Optional[str] = Field(
        default=None,
        description="Media type of the payload (type/subtype).",
    )
    encoding: Optional[str] = Field(
        default=None,
        description="Explicit character encoding hint for textual payloads.",
    )

    @field_validator("inline_text", mode="before")
    @classmethod
    def _coerce_inline_text(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    @field_validator("payload_bytes", mode="before")
    @classmethod
    def _coerce_payload_bytes(cls, value):
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return bytes(value)
        if isinstance(value, str):
            raise ValueError("payload_bytes_string_disallowed")
        raise TypeError("payload_bytes_type")

    @field_validator("payload_base64", mode="before")
    @classmethod
    def _normalize_base64(cls, value):
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("ascii", errors="ignore")
        return str(value).strip()

    @field_validator("object_store_path", mode="before")
    @classmethod
    def _normalize_object_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8", errors="ignore")
        return normalize_optional_string(str(value))

    @field_validator("media_type", mode="before")
    @classmethod
    def _normalize_media_type_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        candidate = str(value).split(";", 1)[0].strip()
        if not candidate:
            return None
        try:
            return normalize_media_type(candidate)
        except ValueError:
            return None

    @field_validator("encoding", mode="before")
    @classmethod
    def _normalize_encoding(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @model_validator(mode="after")
    def _validate_sources(self) -> "DocumentBlobDescriptorV1":
        provided = sum(
            1
            for candidate in (
                self.inline_text,
                self.payload_bytes,
                self.payload_base64,
                self.object_store_path,
            )
            if candidate is not None
        )
        if provided == 0:
            raise ValueError("raw_content_missing")
        if provided > 1:
            raise ValueError("blob_source_ambiguous")
        return self

    @property
    def requires_object_store(self) -> bool:
        return self.object_store_path is not None

    def resolve_payload_bytes(
        self, *, object_store_reader: Optional[Callable[[str], bytes]] = None
    ) -> bytes:
        if self._payload_cache is not None:
            return self._payload_cache
        if self.inline_text is not None:
            payload = self.inline_text.encode("utf-8")
        elif self.payload_bytes is not None:
            payload = self.payload_bytes
        elif self.payload_base64 is not None:
            payload = _decode_base64(self.payload_base64)
        elif self.object_store_path is not None:
            if object_store_reader is None:
                raise ValueError("object_store_reader_required")
            payload = object_store_reader(self.object_store_path)
        else:  # pragma: no cover - defensive guard
            raise ValueError("raw_content_missing")
        self._payload_cache = payload
        return payload

    def payload_size(self, payload_bytes: Optional[bytes] = None) -> int:
        data = (
            payload_bytes if payload_bytes is not None else self.resolve_payload_bytes()
        )
        return len(data)

    def payload_base64_value(self, payload_bytes: Optional[bytes] = None) -> str:
        data = (
            payload_bytes if payload_bytes is not None else self.resolve_payload_bytes()
        )
        return base64.b64encode(data).decode("ascii")


class NormalizedDocumentInputV1(BaseModel):
    """Ingestion contract for raw document payloads."""

    model_config = ConfigDict(
        extra="forbid",
        title="Normalized Document Input v1",
        description="Input contract capturing metadata and payload sources for ingestion.",
    )

    tenant_id: str = Field(
        description="Tenant identifier responsible for the document."
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Workflow identifier grouping the ingestion request.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Channel that supplied the document payload (crawler, upload, …).",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Canonical upstream provider name if known.",
    )
    external_id: Optional[str] = Field(
        default=None,
        description="External identifier assigned by the upstream system.",
    )
    document_id: Any = Field(
        default=None,
        description="Optional stable document identifier supplied by the caller.",
    )
    collection_id: Optional[UUID] = Field(
        default=None,
        description="Optional collection identifier the document belongs to.",
    )
    document_collection_id: Optional[UUID] = Field(
        default=None,
        description="Optional logical DocumentCollection identifier for the document.",
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Optional business case identifier supplied alongside the document.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional set of tags attached to the document for retrieval.",
    )
    origin_uri: Optional[str] = Field(
        default=None,
        description="Original URI describing the payload source.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional human readable title for the document.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional BCP-47 language tag describing the primary language.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque metadata mapping forwarded by the ingestion caller.",
    )
    external_ref: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional provider specific reference attributes.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Optional document version label supplied by the caller.",
    )
    blob: DocumentBlobDescriptorV1 = Field(
        description="Descriptor pointing to the raw payload contents.",
    )

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalize_tenant_id(cls, value: str) -> str:
        return normalize_tenant(value)

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _normalize_workflow(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return normalize_workflow_id(value)

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: Optional[str]) -> Optional[str]:
        normalized = normalize_optional_string(value)
        return normalized.lower() if normalized else None

    @field_validator("provider", "external_id", "case_id", mode="before")
    @classmethod
    def _normalize_optional_identifiers(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("collection_id", mode="before")
    @classmethod
    def _coerce_collection_id(cls, value):
        if value is None:
            return None
        try:
            return _coerce_uuid(value)
        except (TypeError, ValueError):
            return None

    @field_validator("document_collection_id", mode="before")
    @classmethod
    def _coerce_document_collection_id(cls, value):
        if value is None:
            return None
        try:
            return _coerce_uuid(value)
        except (TypeError, ValueError):
            return None

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags_field(cls, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        return normalize_tags(str(item) for item in items)

    @field_validator("origin_uri", mode="before")
    @classmethod
    def _normalize_origin_uri(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("title", mode="before")
    @classmethod
    def _normalize_title_field(cls, value: Optional[str]) -> Optional[str]:
        return normalize_title(value)

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Optional[str]) -> Optional[str]:
        normalized = normalize_optional_string(value)
        if normalized is None:
            return None
        if not is_bcp47_like(normalized):
            raise ValueError("language_invalid")
        return normalized

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value):
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, MutableMapping):
            return dict(value)
        raise TypeError("metadata_mapping_required")

    @field_validator("external_ref", mode="before")
    @classmethod
    def _normalize_external_ref(cls, value):
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("external_ref_mapping_required")
        normalized: Dict[str, str] = {}
        for key, candidate in value.items():
            normalized_key = normalize_optional_string(str(key))
            if not normalized_key:
                continue
            normalized_value = normalize_optional_string(str(candidate))
            if normalized_value is None:
                continue
            normalized[normalized_key] = normalized_value
        return normalized

    @field_validator("version", mode="before")
    @classmethod
    def _normalize_version_field(cls, value: Optional[str]) -> Optional[str]:
        normalized = normalize_optional_string(value)
        if normalized is None:
            return None
        if len(normalized) > 64:
            raise ValueError("version_too_long")
        if not _VERSION_RE.fullmatch(normalized):
            raise ValueError("version_invalid")
        return normalized

    @model_validator(mode="after")
    def _apply_defaults(self) -> "NormalizedDocumentInputV1":
        source = self.source or "crawler"
        self.source = source

        self.workflow_id = resolve_workflow_id(self.workflow_id, required=False)

        metadata_provider = self.metadata.get("provider") if self.metadata else None
        provider = self.provider
        if provider is None and metadata_provider is not None:
            provider = normalize_optional_string(str(metadata_provider))
        if not provider:
            provider = DEFAULT_PROVIDER_BY_SOURCE.get(source) or source
        self.provider = provider

        metadata_external_id = (
            self.metadata.get("external_id") if self.metadata else None
        )
        if self.external_id is None and metadata_external_id is not None:
            self.external_id = normalize_optional_string(str(metadata_external_id))

        if not self.tags:
            metadata_tags = self.metadata.get("tags") if self.metadata else None
            if metadata_tags:
                if isinstance(metadata_tags, (list, tuple, set)):
                    tag_iter = metadata_tags
                else:
                    tag_iter = [metadata_tags]
                self.tags = normalize_tags(str(tag) for tag in tag_iter)

        metadata_origin = self.metadata.get("origin_uri") if self.metadata else None
        if self.origin_uri is None and metadata_origin is not None:
            self.origin_uri = normalize_optional_string(str(metadata_origin))

        metadata_case = self.metadata.get("case_id") if self.metadata else None
        if self.case_id is None and metadata_case is not None:
            self.case_id = normalize_optional_string(str(metadata_case))

        metadata_title = self.metadata.get("title") if self.metadata else None
        if self.title is None and metadata_title is not None:
            self.title = normalize_title(metadata_title)

        metadata_language = self.metadata.get("language") if self.metadata else None
        if self.language is None and metadata_language is not None:
            normalized_language = normalize_optional_string(str(metadata_language))
            if normalized_language and is_bcp47_like(normalized_language):
                self.language = normalized_language

        if self.blob.media_type is None:
            candidate = None
            for key in ("media_type", "content_type", "mime_type"):
                raw_value = self.metadata.get(key)
                if raw_value:
                    candidate = str(raw_value)
                    break
            normalized_media_type = None
            if candidate:
                candidate = candidate.split(";", 1)[0].strip()
                if candidate:
                    try:
                        normalized_media_type = normalize_media_type(candidate)
                    except ValueError:
                        normalized_media_type = None
            if normalized_media_type is None:
                normalized_media_type = "text/plain"
            self.blob = self.blob.model_copy(
                update={"media_type": normalized_media_type}
            )

        if self.blob.encoding is None:
            encoding_hint = None
            for key in ("payload_encoding", "content_encoding"):
                raw_value = self.metadata.get(key)
                if raw_value:
                    encoding_hint = normalize_optional_string(str(raw_value))
                    if encoding_hint:
                        break
            if encoding_hint is None:
                for key in ("media_type", "content_type", "mime_type"):
                    raw_value = self.metadata.get(key)
                    encoding_hint = _extract_charset_param(raw_value)
                    if encoding_hint:
                        break
            if encoding_hint:
                self.blob = self.blob.model_copy(update={"encoding": encoding_hint})

        return self

    @model_validator(mode="after")
    def _validate_collection_consistency(self) -> "NormalizedDocumentInputV1":
        if self.collection_id and self.document_collection_id:
            if self.collection_id != self.document_collection_id:
                raise ValueError("document_collection_mismatch")
        return self

    @property
    def metadata_map(self) -> Mapping[str, Any]:
        return dict(self.metadata)

    @property
    def media_type(self) -> str:
        return self.blob.media_type or "text/plain"

    @property
    def encoding_hint(self) -> Optional[str]:
        if self.blob.encoding:
            return self.blob.encoding
        for key in ("payload_encoding", "content_encoding"):
            raw_value = self.metadata.get(key)
            if raw_value:
                candidate = normalize_optional_string(str(raw_value))
                if candidate:
                    return candidate
        for key in ("media_type", "content_type", "mime_type"):
            raw_value = self.metadata.get(key)
            encoding = _extract_charset_param(str(raw_value)) if raw_value else None
            if encoding:
                return encoding
        return None

    @property
    def requires_object_store(self) -> bool:
        return self.blob.requires_object_store

    @property
    def object_store_path(self) -> Optional[str]:
        return self.blob.object_store_path

    def resolve_payload_bytes(
        self, *, object_store_reader: Optional[Callable[[str], bytes]] = None
    ) -> bytes:
        return self.blob.resolve_payload_bytes(object_store_reader=object_store_reader)

    def resolve_payload_text(self, payload_bytes: Optional[bytes] = None) -> str:
        if self.blob.inline_text is not None:
            return self.blob.inline_text
        payload = payload_bytes
        if payload is None:
            payload = self.resolve_payload_bytes()
        encoding_hint = self.encoding_hint
        if encoding_hint:
            try:
                return payload.decode(encoding_hint)
            except (LookupError, UnicodeDecodeError):
                pass
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return payload.decode("utf-8", errors="replace")

    def compute_checksum(self, payload_bytes: Optional[bytes] = None) -> str:
        payload = (
            payload_bytes if payload_bytes is not None else self.resolve_payload_bytes()
        )
        return hashlib.sha256(payload).hexdigest()

    def payload_base64(self, payload_bytes: Optional[bytes] = None) -> str:
        return self.blob.payload_base64_value(payload_bytes)

    def payload_size(self, payload_bytes: Optional[bytes] = None) -> int:
        return self.blob.payload_size(payload_bytes)

    def build_external_reference(self, document_id: Optional[UUID]) -> Dict[str, str]:
        reference = dict(self.external_ref)
        provider = self.provider or self.source or "unknown"
        reference.setdefault("provider", provider)
        if self.external_id:
            reference.setdefault("external_id", self.external_id)
        elif document_id is not None:
            reference.setdefault("external_id", f"{provider}:{document_id}")
        if self.case_id:
            reference.setdefault("case_id", self.case_id)
        return reference

    @property
    def metadata_payload(self) -> Mapping[str, Any]:
        payload = {
            "tenant_id": self.tenant_id,
            "workflow_id": self.workflow_id,
            "case_id": self.case_id,
            "provider": self.provider,
            "origin_uri": self.origin_uri,
            "source": self.source,
        }
        return dict({k: v for k, v in payload.items() if v is not None})

    @classmethod
    def from_raw(
        cls,
        *,
        raw_reference: Mapping[str, Any],
        tenant_id: str,
        case_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> "NormalizedDocumentInputV1":
        metadata_raw = raw_reference.get("metadata")
        if metadata_raw is None:
            metadata: Dict[str, Any] = {}
        elif isinstance(metadata_raw, Mapping):
            metadata = dict(metadata_raw)
        elif isinstance(metadata_raw, MutableMapping):
            metadata = dict(metadata_raw)
        else:
            raise TypeError("metadata_mapping_required")

        blob_media_type = (
            raw_reference.get("media_type")
            or raw_reference.get("content_type")
            or raw_reference.get("mime_type")
            or metadata.get("media_type")
            or metadata.get("content_type")
            or metadata.get("mime_type")
        )
        blob_encoding = (
            raw_reference.get("payload_encoding")
            or raw_reference.get("content_encoding")
            or metadata.get("payload_encoding")
            or metadata.get("content_encoding")
        )

        payload_base64 = raw_reference.get("payload_base64")
        if isinstance(payload_base64, (bytes, bytearray, memoryview)):
            payload_base64 = bytes(payload_base64).decode("ascii", errors="ignore")

        object_path = raw_reference.get("payload_path")
        if object_path is not None and not isinstance(object_path, str):
            object_path = str(object_path)

        descriptor = DocumentBlobDescriptorV1(
            inline_text=raw_reference.get("content"),
            payload_bytes=raw_reference.get("payload_bytes"),
            payload_base64=payload_base64,
            object_store_path=object_path,
            media_type=blob_media_type,
            encoding=blob_encoding,
        )

        provider = (
            metadata.get("provider")
            or raw_reference.get("provider")
            or DEFAULT_PROVIDER_BY_SOURCE.get(source or metadata.get("source", ""))
        )

        external_id = metadata.get("external_id") or raw_reference.get("external_id")

        tags = metadata.get("tags") or raw_reference.get("tags")

        origin_uri = metadata.get("origin_uri") or raw_reference.get("origin_uri")

        title = metadata.get("title") or raw_reference.get("title")

        language = metadata.get("language") or raw_reference.get("language")

        external_ref = metadata.get("external_ref") or raw_reference.get("external_ref")

        version = metadata.get("version") or raw_reference.get("version")

        return cls(
            tenant_id=tenant_id,
            workflow_id=(
                workflow_id
                or metadata.get("workflow_id")
                or raw_reference.get("workflow_id")
            ),
            source=(source or metadata.get("source") or raw_reference.get("source")),
            provider=provider,
            external_id=external_id,
            document_id=raw_reference.get("document_id"),
            collection_id=metadata.get("collection_id"),
            case_id=case_id or metadata.get("case_id") or raw_reference.get("case_id"),
            tags=tags,
            origin_uri=origin_uri,
            title=title,
            language=language,
            metadata=metadata,
            external_ref=external_ref,
            version=version,
            blob=descriptor,
        )


class Asset(BaseModel):
    """Representation of an extracted multimodal asset."""

    model_config = ConfigDict(
        extra="forbid",
        title="Document Asset",
        description="Describes an extracted asset including its storage blob and context.",
    )

    ref: AssetRef = Field(description="Reference identifiers for the asset.")
    media_type: str = Field(
        description=(
            "Media type describing the asset content (lowercase type/subtype, no parameters)."
        )
    )
    blob: BlobLocator = Field(
        description="Blob containing the asset payload or pointer."
    )
    origin_uri: Optional[str] = Field(
        default=None, description="Optional URI of the source that produced the asset."
    )
    page_index: Optional[int] = Field(
        default=None, description="Optional page index the asset originates from."
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Normalized bounding box [x0, y0, x1, y1] within the page.",
    )
    context_before: Optional[str] = Field(
        default=None,
        description="Optional textual context preceding the asset (<= 2048 bytes).",
    )
    context_after: Optional[str] = Field(
        default=None,
        description="Optional textual context following the asset (<= 2048 bytes).",
    )
    parent_ref: Optional[str] = Field(
        default=None,
        description="Logical reference to the parent block within the source document.",
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Optional OCR extracted text for the asset (<= 8192 bytes).",
    )
    text_description: Optional[str] = Field(
        default=None,
        description="Optional textual description for the asset (<= 2048 bytes).",
    )
    caption_source: CaptionSource = Field(
        default="none",
        description="Source that produced the current text description (alt_text, vlm, ocr, …).",
    )
    caption_method: Literal["vlm_caption", "ocr_only", "manual", "none"] = Field(
        description="Method used to derive the text description or caption."
    )
    caption_model: Optional[str] = Field(
        default=None,
        description="Model identifier if captions were generated by a VLM.",
    )
    caption_confidence: Optional[float] = Field(
        default=None,
        description="Confidence for model generated captions (0.0-1.0).",
    )
    perceptual_hash: Optional[str] = Field(
        default=None,
        description="Optional 64-bit perceptual hash for image deduplication.",
    )
    asset_kind: Optional[str] = Field(
        default=None,
        description="Optional semantic classification of the asset (e.g. chart_source).",
    )
    created_at: datetime = Field(
        description="UTC timestamp when the asset was generated or ingested."
    )
    checksum: str = Field(
        description="Checksum ensuring integrity of the asset payload."
    )

    @property
    def workflow_id(self) -> str:
        """Expose the workflow identifier of the nested reference."""

        return self.ref.workflow_id

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str) -> str:
        return normalize_media_type(value)

    @field_validator("origin_uri", mode="before")
    @classmethod
    def _normalize_origin(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return None
        return validate_bbox(list(value))

    @field_validator("context_before")
    @classmethod
    def _limit_context_before(cls, value: Optional[str]) -> Optional[str]:
        return truncate_text(value, 2048)

    @field_validator("context_after")
    @classmethod
    def _limit_context_after(cls, value: Optional[str]) -> Optional[str]:
        return truncate_text(value, 2048)

    @field_validator("parent_ref", mode="before")
    @classmethod
    def _normalize_parent_ref(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("ocr_text")
    @classmethod
    def _limit_ocr_text(cls, value: Optional[str]) -> Optional[str]:
        return truncate_text(value, 8192)

    @field_validator("text_description")
    @classmethod
    def _limit_text_description(cls, value: Optional[str]) -> Optional[str]:
        return truncate_text(value, 2048)

    @field_validator("caption_source", mode="before")
    @classmethod
    def _normalize_caption_source(cls, value: Optional[str]) -> str:
        candidate = normalize_optional_string(value) or "none"
        candidate = candidate.lower()
        if candidate not in _CAPTION_SOURCES:
            raise ValueError("caption_source_invalid")
        return candidate

    @field_validator("caption_model", mode="before")
    @classmethod
    def _normalize_caption_model(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("caption_confidence")
    @classmethod
    def _validate_caption_confidence(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if value < 0.0 or value > 1.0:
            raise ValueError("caption_confidence_range")
        return value

    @field_validator("perceptual_hash", mode="before")
    @classmethod
    def _validate_perceptual_hash(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        candidate = normalize_optional_string(value)
        if candidate is None:
            return None
        lowered = candidate.lower()
        if len(lowered) != 16 or any(ch not in "0123456789abcdef" for ch in lowered):
            raise ValueError("perceptual_hash_invalid")
        return lowered

    @field_validator("asset_kind", mode="before")
    @classmethod
    def _normalize_asset_kind(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_string(value)

    @field_validator("page_index")
    @classmethod
    def _validate_page_index(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 0:
            raise ValueError("page_index_negative")
        return value

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at_naive")
        return value.astimezone(timezone.utc)

    @field_validator("checksum")
    @classmethod
    def _validate_checksum(cls, value: str) -> str:
        if not _HEX_64_RE.fullmatch(value):
            raise ValueError("checksum_invalid")
        return value

    @model_validator(mode="after")
    def _check_media_consistency(self) -> "Asset":
        if isinstance(self.blob, InlineBlob):
            if self.media_type.lower() != self.blob.media_type.lower():
                raise ValueError("media_type_mismatch")
        return self

    @model_validator(mode="after")
    def _check_caption_requirements(self) -> "Asset":
        if self.caption_method == "vlm_caption":
            if not self.caption_model:
                raise ValueError("caption_model_required")
            if self.caption_confidence is None:
                raise ValueError("caption_confidence_required")
        return self

    @model_validator(mode="after")
    def _check_checksum(self) -> "Asset":
        if not is_strict_checksums_enabled():
            return self
        blob_sha = getattr(self.blob, "sha256", None)
        if not blob_sha:
            raise ValueError("asset_checksum_missing")
        if blob_sha != self.checksum:
            raise ValueError("asset_checksum_mismatch")
        return self

    @model_validator(mode="after")
    def _enforce_media_guard(self) -> "Asset":
        guard = get_asset_media_guard()
        if guard and not isinstance(self.blob, InlineBlob):
            if not any(self.media_type.startswith(prefix) for prefix in guard):
                raise ValueError("media_type_guard")
        return self


class NormalizedDocument(BaseModel):
    """Complete normalized document with metadata and optional assets."""

    model_config = ConfigDict(
        extra="forbid",
        title="Normalized Document",
        description="Represents a stored document with metadata, blob and derived assets.",
    )

    ref: DocumentRef = Field(description="Identifier reference for the document.")
    meta: DocumentMeta = Field(description="Associated metadata for the document.")
    blob: BlobLocator = Field(description="Primary blob storing the document contents.")
    checksum: str = Field(description="Checksum for the main document blob.")
    created_at: datetime = Field(
        description="UTC timestamp when the document was ingested into the system."
    )
    source: Optional[Literal["upload", "crawler", "integration", "other"]] = Field(
        default=None,
        description=(
            "Ingestion channel describing how the document entered the platform "
            "(e.g. `crawler`, `upload`). This is distinct from "
            "`meta.external_ref['provider']`, which names the upstream system or "
            "integration."
        ),
    )
    lifecycle_state: Literal["active", "retired", "deleted"] = Field(
        default="active",
        description="Lifecycle status indicating whether the document is active, retired or deleted.",
    )
    assets: List[Asset] = Field(
        default_factory=list,
        description="Assets that were extracted from the document.",
    )
    content_normalized: Optional[str] = Field(
        default=None,
        description="Normalized textual content of the document.",
    )
    primary_text: Optional[str] = Field(
        default=None,
        description="Legacy primary text field.",
    )

    @field_validator("checksum")
    @classmethod
    def _validate_checksum(cls, value: str) -> str:
        if not _HEX_64_RE.fullmatch(value):
            raise ValueError("checksum_invalid")
        return value

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at_naive")
        return value.astimezone(timezone.utc)

    @field_validator("lifecycle_state", mode="before")
    @classmethod
    def _normalize_lifecycle_state(cls, value: Optional[str]) -> str:
        if value is None:
            return "active"
        candidate = str(value).strip().lower()
        return candidate or "active"

    @model_validator(mode="after")
    def _validate_relationships(self) -> "NormalizedDocument":
        if self.meta.tenant_id != self.ref.tenant_id:
            raise ValueError("meta_tenant_mismatch")
        if self.meta.workflow_id != self.ref.workflow_id:
            raise ValueError("meta_workflow_mismatch")
        for asset in self.assets:
            if asset.ref.tenant_id != self.ref.tenant_id:
                raise ValueError("asset_tenant_mismatch")
            if asset.ref.document_id != self.ref.document_id:
                raise ValueError("asset_document_mismatch")
            if asset.ref.collection_id != self.ref.collection_id:
                raise ValueError("asset_collection_mismatch")
            if asset.ref.workflow_id != self.ref.workflow_id:
                raise ValueError("asset_workflow_mismatch")
        if is_strict_checksums_enabled():
            blob_sha = getattr(self.blob, "sha256", None)
            if not blob_sha:
                raise ValueError("document_checksum_missing")
            if blob_sha != self.checksum:
                raise ValueError("document_checksum_mismatch")
        if self.source == "integration":
            external_ref = self.meta.external_ref or {}
            if any(key in external_ref for key in ("model", "generated_at", "topic")):
                required_keys = ("provider", "model", "generated_at", "topic")
                for key in required_keys:
                    if key not in external_ref:
                        raise ValueError(f"llm_external_ref_missing_{key}")
                generated_at = external_ref["generated_at"]
                iso_value = (
                    generated_at[:-1] + "+00:00"
                    if generated_at.endswith("Z")
                    else generated_at
                )
                try:
                    datetime.fromisoformat(iso_value)
                except ValueError as exc:  # pragma: no cover - invalid ISO format
                    raise ValueError("llm_generated_at_invalid") from exc
                if self.meta.origin_uri is not None:
                    raise ValueError("llm_origin_present")
                if self.meta.crawl_timestamp is not None:
                    raise ValueError("llm_crawl_timestamp_present")
        return self


def document_ref_schema() -> Dict[str, object]:
    """Return the JSON schema for :class:`DocumentRef`."""

    return DocumentRef.model_json_schema()


def blob_locator_schema() -> Dict[str, object]:
    """Return the JSON schema for :class:`BlobLocator`."""

    return TypeAdapter(BlobLocator).json_schema()


def normalized_document_schema() -> Dict[str, object]:
    """Return the JSON schema for :class:`NormalizedDocument`."""

    return NormalizedDocument.model_json_schema()


def document_meta_schema() -> Dict[str, object]:
    """Return the JSON schema for :class:`DocumentMeta`."""

    return DocumentMeta.model_json_schema()


def asset_ref_schema() -> Dict[str, object]:
    """Return the JSON schema for :class:`AssetRef`."""

    return AssetRef.model_json_schema()


def asset_schema() -> Dict[str, object]:
    """Return the JSON schema for :class:`Asset`."""

    return Asset.model_json_schema()
