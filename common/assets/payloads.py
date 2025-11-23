"""Asset ingestion payload definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = ["AssetIngestPayload"]


@dataclass(frozen=True)
class AssetIngestPayload:
    """Neutral representation of an asset ready for ingestion."""

    media_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    content: Optional[bytes] = None
    file_uri: Optional[str] = None
    page_index: Optional[int] = None
    bbox: Optional[Tuple[float, ...]] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        meta = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "metadata", meta)
        bbox = self._coerce_bbox(self.bbox)
        object.__setattr__(self, "bbox", bbox)
        if self.content is None and self.file_uri is None:
            raise ValueError("asset_ingest_location")

    @staticmethod
    def _coerce_bbox(bbox: Optional[Sequence[float]]) -> Optional[Tuple[float, ...]]:
        if bbox is None:
            return None
        return tuple(float(value) for value in bbox)
