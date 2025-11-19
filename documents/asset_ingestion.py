"""Asset ingestion components with separated concerns.

Refactored from _persist_asset god-function into:
- CaptionResolver: Caption prioritization strategy
- AssetDeduplicator: Checksum-based deduplication
- BlobStorageAdapter: Storage abstraction
- AssetIngestionPipeline: Orchestration
"""

import hashlib
import io
from dataclasses import dataclass
from typing import Optional, Any, Protocol, Sequence, Tuple, Mapping
from uuid import UUID, uuid5

from PIL import Image, ImageOps

from documents.contract_utils import (
    normalize_string,
    normalize_optional_string,
    is_image_mediatype,
)
from .parsers import ParsedAsset


# Caption source confidence mapping (from pipeline.py)
_CAPTION_SOURCE_CONFIDENCE: dict[str, float] = {
    "alt_text": 1.0,
    "figure_caption": 0.9,
    "notes": 0.8,
    "context_after": 0.7,
    "context_before": 0.6,
    "origin": 0.5,
}


@dataclass(frozen=True)
class CaptionResult:
    """Resolved caption with metadata."""

    text: Optional[str]
    source: str
    method: str  # "manual" or "none"
    confidence: Optional[float]


class CaptionResolver:
    """Strategy for resolving image captions from multiple sources.

    Priority:
    1. caption_candidates (list of [source, text] tuples)
    2. caption_source + caption_text (direct fields)
    3. "none" (no caption available)
    """

    def __init__(self, confidence_map: Optional[dict[str, float]] = None):
        """Initialize with optional custom confidence mapping.

        Args:
            confidence_map: Map from caption source to confidence [0.0, 1.0]
        """
        self._confidence_map = confidence_map or _CAPTION_SOURCE_CONFIDENCE

    def resolve(self, metadata_payload: Mapping[str, Any]) -> CaptionResult:
        """Resolve caption from metadata payload.

        Args:
            metadata_payload: Parsed asset metadata dictionary

        Returns:
            CaptionResult with highest-priority caption

        Business logic:
        - Prioritizes candidates list over direct fields
        - Uses first valid candidate (fail-fast)
        - Falls back to "none" if no valid captions
        """
        # Strategy 1: caption_candidates list
        raw_candidates = metadata_payload.get("caption_candidates")
        if isinstance(raw_candidates, Sequence):
            for entry in raw_candidates:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    continue
                raw_source, raw_text = entry
                source = (
                    normalize_string(str(raw_source)) if raw_source is not None else ""
                )
                if not source:
                    continue
                if raw_text is None:
                    continue
                text = normalize_optional_string(str(raw_text))
                if text:
                    return CaptionResult(
                        text=text,
                        source=source,
                        method="manual",
                        confidence=self._confidence_map.get(source),
                    )

        # Strategy 2: Direct caption_source + caption_text fields
        raw_source = metadata_payload.get("caption_source")
        raw_text = metadata_payload.get("caption_text")
        source = normalize_string(str(raw_source)) if raw_source is not None else ""
        text = (
            normalize_optional_string(str(raw_text)) if raw_text is not None else None
        )
        if source and text:
            return CaptionResult(
                text=text,
                source=source,
                method="manual",
                confidence=self._confidence_map.get(source),
            )

        # Fallback: No caption
        return CaptionResult(text=None, source="none", method="none", confidence=None)


@dataclass(frozen=True)
class BlobDescriptor:
    """Descriptor for stored blob (file or external URI)."""

    checksum: str
    payload_dict: dict[str, Any]  # Serializable blob metadata
    perceptual_hash: Optional[str] = None


class StorageProtocol(Protocol):
    """Duck-typed protocol for storage backends."""

    def put(self, payload: bytes) -> Tuple[str, str, int]:
        """Store bytes, return (uri, checksum, size)."""
        ...

    def get(self, uri: str) -> bytes:
        """Retrieve bytes by URI."""
        ...


class BlobStorageAdapter:
    """Adapter for blob storage with hash computation.

    Handles:
    - Binary content storage
    - External URI references
    - Perceptual hashing for images
    """

    def __init__(self, storage: StorageProtocol):
        """Initialize with storage backend.

        Args:
            storage: Storage backend (duck-typed)
        """
        self._storage = storage

    def store_content(self, content: bytes, media_type: str) -> BlobDescriptor:
        """Store binary content and compute hashes.

        Args:
            content: Binary payload
            media_type: MIME type for perceptual hash detection

        Returns:
            BlobDescriptor with checksum and storage metadata
        """
        checksum = hashlib.sha256(content).hexdigest()
        uri, storage_checksum, size = self._storage.put(content)

        if storage_checksum != checksum:
            raise ValueError("asset_checksum_mismatch")

        perceptual_hash = (
            self._compute_perceptual_hash(content)
            if is_image_mediatype(media_type)
            else None
        )

        payload_dict = {
            "type": "file",
            "uri": uri,
            "sha256": storage_checksum,
            "size": size,
        }

        return BlobDescriptor(
            checksum=checksum,
            payload_dict=payload_dict,
            perceptual_hash=perceptual_hash,
        )

    def store_file_uri(self, file_uri: str, media_type: str) -> BlobDescriptor:
        """Store reference to existing file URI.

        Args:
            file_uri: URI of existing file
            media_type: MIME type for perceptual hash detection

        Returns:
            BlobDescriptor with checksum (from content or URI hash)
        """
        try:
            payload = self._storage.get(file_uri)
        except (KeyError, ValueError):
            payload = None

        if payload is not None:
            # File exists, compute content hash
            checksum = hashlib.sha256(payload).hexdigest()
            perceptual_hash = (
                self._compute_perceptual_hash(payload)
                if is_image_mediatype(media_type)
                else None
            )
            payload_dict = {
                "type": "file",
                "uri": file_uri,
                "sha256": checksum,
                "size": len(payload),
            }
        else:
            # External URI, hash the URI itself
            checksum = hashlib.sha256(file_uri.encode("utf-8")).hexdigest()
            perceptual_hash = None
            from urllib.parse import urlparse

            parsed = urlparse(file_uri)
            scheme = parsed.scheme.lower()
            kind = scheme if scheme in {"http", "https", "s3", "gcs"} else "http"

            payload_dict = {
                "type": "external",
                "kind": kind,
                "uri": file_uri,
            }

        return BlobDescriptor(
            checksum=checksum,
            payload_dict=payload_dict,
            perceptual_hash=perceptual_hash,
        )

    @staticmethod
    def _compute_perceptual_hash(payload: bytes) -> Optional[str]:
        """Compute average hash for image payload.

        Args:
            payload: Binary image data

        Returns:
            16-character hex hash or None if not an image
        """
        try:
            with Image.open(io.BytesIO(payload)) as img:
                grey = ImageOps.grayscale(img)
                resample_attr = getattr(Image, "Resampling", None)
                resample = getattr(
                    resample_attr, "LANCZOS", getattr(Image, "LANCZOS", 1)
                )
                resized = grey.resize((8, 8), resample=resample)
                pixels = list(resized.getdata())
        except Exception:  # pragma: no cover - fall back on unsupported formats
            return None

        if not pixels:
            return None

        average = sum(pixels) / len(pixels)
        bits = 0
        for value in pixels:
            bits = (bits << 1) | int(value >= average)

        return f"{bits:016x}"


class RepositoryProtocol(Protocol):
    """Duck-typed protocol for asset repositories."""

    def get_asset(
        self, tenant_id: str, asset_id: UUID, workflow_id: str
    ) -> Optional[Any]:
        """Get existing asset by ID."""
        ...

    def add_asset(self, asset_obj: Any, workflow_id: str) -> Any:
        """Add new asset."""
        ...


@dataclass(frozen=True)
class AssetMetadata:
    """Extracted metadata for asset ingestion."""

    locator: str
    parent_ref: Optional[str]
    asset_kind: Optional[str]
    origin_uri: Optional[str]
    raw_metadata: Mapping[str, Any]


class AssetIngestionPipeline:
    """Orchestrates asset persistence with separated concerns.

    Replaces the _persist_asset god-function with:
    1. Metadata extraction
    2. Caption resolution (CaptionResolver)
    3. Blob storage (BlobStorageAdapter)
    4. Deduplication check (AssetDeduplicator)
    5. Repository persistence
    """

    def __init__(
        self,
        repository: RepositoryProtocol,
        storage: StorageProtocol,
        contracts: Any,
        caption_resolver: Optional[CaptionResolver] = None,
    ):
        """Initialize pipeline components.

        Args:
            repository: Asset repository
            storage: Blob storage backend
            contracts: Document contracts factory
            caption_resolver: Optional custom caption resolver
        """
        self._repository = repository
        self._storage = storage
        self._contracts = contracts
        self._caption_resolver = caption_resolver or CaptionResolver()
        self._blob_adapter = BlobStorageAdapter(storage)

    def persist_asset(
        self,
        index: int,
        parsed_asset: ParsedAsset,
        tenant_id: str,
        workflow_id: str,
        document_id: UUID,
        collection_id: str,
        created_at: Any,
    ) -> Any:
        """Persist parsed asset with deduplication.

        Args:
            index: Asset index in document
            parsed_asset: Parsed asset data
            tenant_id: Tenant ID
            workflow_id: Workflow ID
            document_id: Parent document ID
            collection_id: Collection ID
            created_at: Creation timestamp

        Returns:
            Stored asset object from repository
        """
        # 1. Extract metadata
        asset_meta = self._extract_metadata(index, parsed_asset)

        # 2. Resolve caption
        caption_result = self._caption_resolver.resolve(asset_meta.raw_metadata)

        # 3. Store blob & compute checksums
        blob_desc = self._store_blob(parsed_asset)

        # 4. Generate deterministic asset ID
        asset_id = uuid5(document_id, f"asset:{asset_meta.locator}")

        # 5. Check for existing asset (deduplication)
        existing = self._repository.get_asset(
            tenant_id, asset_id, workflow_id=workflow_id
        )
        if existing is not None and existing.checksum == blob_desc.checksum:
            return existing

        # 6. Build asset object
        asset_ref = self._contracts.asset_ref(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            asset_id=asset_id,
            document_id=document_id,
            collection_id=collection_id,
        )

        bbox = list(parsed_asset.bbox) if parsed_asset.bbox is not None else None

        # Resolve origin_uri: prefer metadata, fallback to file_uri for external refs
        origin_uri = asset_meta.origin_uri
        if origin_uri is None and blob_desc.payload_dict.get("type") == "external":
            origin_uri = blob_desc.payload_dict.get("uri")

        asset_obj = self._contracts.asset(
            ref=asset_ref,
            media_type=parsed_asset.media_type,
            blob=blob_desc.payload_dict,
            origin_uri=origin_uri,
            page_index=parsed_asset.page_index,
            bbox=bbox,
            context_before=parsed_asset.context_before,
            context_after=parsed_asset.context_after,
            ocr_text=None,
            text_description=caption_result.text,
            caption_method=caption_result.method,
            caption_model=None,
            caption_confidence=caption_result.confidence,
            caption_source=caption_result.source,
            parent_ref=asset_meta.parent_ref,
            perceptual_hash=blob_desc.perceptual_hash,
            asset_kind=asset_meta.asset_kind,
            created_at=created_at,
            checksum=blob_desc.checksum,
        )

        # 7. Persist to repository
        stored = self._repository.add_asset(asset_obj, workflow_id=workflow_id)
        return stored

    def _extract_metadata(self, index: int, parsed_asset: ParsedAsset) -> AssetMetadata:
        """Extract and normalize asset metadata.

        Args:
            index: Asset index (for fallback locator)
            parsed_asset: Parsed asset

        Returns:
            AssetMetadata with normalized fields
        """
        metadata_payload_raw = getattr(parsed_asset, "metadata", None)
        if isinstance(metadata_payload_raw, Mapping):
            metadata_payload = dict(metadata_payload_raw)
        else:
            metadata_payload = {}

        # Locator
        raw_locator = metadata_payload.get("locator")
        locator = normalize_string(str(raw_locator)) if raw_locator is not None else ""
        if not locator:
            locator = f"asset-index:{index}"

        # Parent ref
        raw_parent_ref = metadata_payload.get("parent_ref")
        parent_ref = (
            normalize_optional_string(str(raw_parent_ref))
            if raw_parent_ref is not None
            else None
        )

        # Asset kind
        raw_kind = metadata_payload.get("asset_kind")
        asset_kind = (
            normalize_optional_string(str(raw_kind)) if raw_kind is not None else None
        )

        # Origin URI
        raw_origin = metadata_payload.get("origin_uri")
        origin_uri = (
            normalize_optional_string(str(raw_origin))
            if raw_origin is not None
            else None
        )

        return AssetMetadata(
            locator=locator,
            parent_ref=parent_ref,
            asset_kind=asset_kind,
            origin_uri=origin_uri,
            raw_metadata=metadata_payload,
        )

    def _store_blob(self, parsed_asset: ParsedAsset) -> BlobDescriptor:
        """Store blob content or reference.

        Args:
            parsed_asset: Parsed asset

        Returns:
            BlobDescriptor with checksum and payload metadata

        Raises:
            ValueError: If neither content nor file_uri is available
        """
        if parsed_asset.content is not None:
            # Binary content available
            payload = bytes(parsed_asset.content)
            return self._blob_adapter.store_content(payload, parsed_asset.media_type)
        elif parsed_asset.file_uri is not None:
            # File URI reference
            return self._blob_adapter.store_file_uri(
                parsed_asset.file_uri, parsed_asset.media_type
            )
        else:
            raise ValueError("parsed_asset_location")
