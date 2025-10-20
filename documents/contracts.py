"""Contracts for normalized documents and assets."""

from __future__ import annotations

import base64
import hashlib
import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import (
    Annotated,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Any,
    Mapping,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
    TypeAdapter,
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
)


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


_STRICT_CHECKSUMS: ContextVar[bool] = ContextVar("STRICT_CHECKSUMS", default=False)
_ASSET_MEDIA_GUARD: ContextVar[Optional[Tuple[str, ...]]] = ContextVar(
    "ASSET_MEDIA_GUARD", default=None
)


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover - propagate message
        raise ValueError("base64_invalid") from exc


def set_strict_checksums(enabled: bool = True) -> None:
    """Toggle strict checksum verification for contract models."""

    _STRICT_CHECKSUMS.set(bool(enabled))


@contextmanager
def strict_checksums(enabled: bool = True):
    """Context manager enabling strict checksum validation within the scope."""

    token = _STRICT_CHECKSUMS.set(bool(enabled))
    try:
        yield
    finally:
        _STRICT_CHECKSUMS.reset(token)


def _normalize_guard_prefix(value: str) -> str:
    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("media_guard_empty")
    return normalized.lower()


def set_asset_media_guard(prefixes: Optional[Iterable[str]]) -> None:
    """Configure allowed media type prefixes for non-inline asset blobs."""

    normalized = None
    if prefixes:
        normalized = tuple(_normalize_guard_prefix(prefix) for prefix in prefixes)
    _ASSET_MEDIA_GUARD.set(normalized)


@contextmanager
def asset_media_guard(prefixes: Optional[Iterable[str]]):
    """Context manager enforcing non-inline asset media type prefixes.

    Inline blobs continue to require an exact media type match and therefore do
    not participate in the prefix guard logic.
    """

    if prefixes:
        normalized = tuple(_normalize_guard_prefix(prefix) for prefix in prefixes)
    else:
        normalized = None
    token = _ASSET_MEDIA_GUARD.set(normalized)
    try:
        yield
    finally:
        _ASSET_MEDIA_GUARD.reset(token)


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

    @field_validator("document_id", "collection_id", mode="before")
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


class InlineBlob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _decoded_payload: Optional[bytes] = PrivateAttr(default=None)

    type: Literal["inline"] = Field(
        description="Discriminator identifying inline data blobs."
    )
    media_type: str = Field(
        description=(
            "RFC compliant media type without parameters (type/subtype lowercase)."
        )
    )
    base64: str = Field(description="Base64 encoded payload contents.")
    sha256: str = Field(description="Hex encoded SHA-256 checksum of the payload.")
    size: int = Field(description="Payload size in bytes.")

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str) -> str:
        return normalize_media_type(value)

    @field_validator("base64")
    @classmethod
    def _validate_base64(cls, value: str) -> str:
        return value.strip()

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
    def _verify_payload(self) -> "InlineBlob":
        decoded = _decode_base64(self.base64)
        self._decoded_payload = decoded
        if len(decoded) != self.size:
            raise ValueError("inline_size_mismatch")
        if _STRICT_CHECKSUMS.get():
            digest = hashlib.sha256(decoded).hexdigest()
            if digest != self.sha256:
                raise ValueError("inline_checksum_mismatch")
        return self

    def decoded_payload(self) -> bytes:
        """Return the cached inline payload, decoding at most once."""
        if self._decoded_payload is None:
            self._decoded_payload = _decode_base64(self.base64)
        return self._decoded_payload


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


BlobLocator = Annotated[
    Union[FileBlob, InlineBlob, ExternalBlob],
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
            "(<= 16 entries; keys <= 128 chars, values <= 512 chars; limits enforced)."
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
        if not _STRICT_CHECKSUMS.get():
            return self
        blob_sha = getattr(self.blob, "sha256", None)
        if not blob_sha:
            raise ValueError("asset_checksum_missing")
        if blob_sha != self.checksum:
            raise ValueError("asset_checksum_mismatch")
        return self

    @model_validator(mode="after")
    def _enforce_media_guard(self) -> "Asset":
        guard = _ASSET_MEDIA_GUARD.get()
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
        description="Source channel through which the document entered the system.",
    )
    assets: List[Asset] = Field(
        default_factory=list,
        description="Assets that were extracted from the document.",
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
        if _STRICT_CHECKSUMS.get():
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
