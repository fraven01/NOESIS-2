from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from common.logging import get_logger

from ai_core.infra import object_store
from ai_core.segmentation import segment_markdown_blocks


logger = get_logger(__name__)


@dataclass
class StructuredDocument:
    blocks: List[Dict[str, object]]
    fallback_segments: List[str]


class StructuredBlockReader:
    """Load structured blocks from the object store when available."""

    def __init__(
        self,
        store=object_store,
        segmenter=segment_markdown_blocks,
    ) -> None:
        self._store = store
        self._segmenter = segmenter

    def read(self, meta: Mapping[str, Any], text: str) -> StructuredDocument:
        path_value = meta.get("parsed_blocks_path")
        structured_blocks: List[Dict[str, object]] = []
        if path_value:
            try:
                payload = self._store.read_json(str(path_value))
            except FileNotFoundError:
                payload = None
            except Exception:
                logger.warning(
                    "ingestion.chunk.blocks_read_failed",
                    extra={
                        "tenant_id": meta.get("tenant_id"),
                        "case_id": meta.get("case_id"),
                        "path": path_value,
                    },
                )
                payload = None
            if isinstance(payload, dict):
                raw_blocks = payload.get("blocks") or []
                if isinstance(raw_blocks, list):
                    structured_blocks = [
                        entry for entry in raw_blocks if isinstance(entry, dict)
                    ]

        fallback_segments: List[str] = []
        if not structured_blocks:
            fallback_segments = self._segmenter(text)

        return StructuredDocument(
            blocks=structured_blocks,
            fallback_segments=fallback_segments,
        )
