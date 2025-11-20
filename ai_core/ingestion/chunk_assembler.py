from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ai_core.ingestion.parent_capture import ChunkCandidate


@dataclass
class ChunkAssemblerInput:
    prefix: str
    chunk_candidates: Sequence[ChunkCandidate]
    hard_limit: int
    meta: Mapping[str, Any]
    text_path: str
    content_hash: str
    document_id: Optional[str]


class ChunkAssembler:
    """Build finalized chunk payloads from chunk candidates."""

    def __init__(
        self,
        *,
        token_counter: Callable[[str], int],
        split_by_limit: Callable[[str, int], List[str]],
        normalizer: Callable[[str], str],
    ) -> None:
        self._token_counter = token_counter
        self._split_by_limit = split_by_limit
        self._normalizer = normalizer

    def assemble(self, data: ChunkAssemblerInput) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        chunk_index = 0
        for candidate in data.chunk_candidates:
            prefix_parts = [part for part in (data.prefix, candidate.heading_prefix) if part]
            prefix_token_count = sum(
                self._token_counter(part) for part in prefix_parts if part
            )
            body_limit = data.hard_limit - prefix_token_count
            if prefix_token_count >= data.hard_limit:
                body_limit = 0
            body_segments: List[str] = [candidate.body]
            if body_limit > 0:
                adjusted_segments: List[str] = []
                for segment in body_segments:
                    if self._token_counter(segment) > body_limit:
                        adjusted_segments.extend(self._split_by_limit(segment, body_limit))
                    else:
                        adjusted_segments.append(segment)
                body_segments = adjusted_segments or [""]
            elif not candidate.body:
                body_segments = [""]

            for segment in body_segments:
                chunk_text = segment
                if prefix_parts:
                    combined_prefix = "\n\n".join(
                        str(part).strip() for part in prefix_parts if str(part).strip()
                    )
                    if combined_prefix:
                        chunk_text = (
                            f"{combined_prefix}\n\n{chunk_text}" if chunk_text else combined_prefix
                        )
                normalized = self._normalizer(chunk_text)
                chunk_hash_input = f"{data.content_hash}:{chunk_index}".encode("utf-8")
                chunk_hash = hashlib.sha256(chunk_hash_input).hexdigest()
                chunk_meta: Dict[str, Any] = {
                    "tenant_id": data.meta["tenant_id"],
                    "case_id": data.meta.get("case_id"),
                    "source": data.text_path,
                    "hash": chunk_hash,
                    "external_id": data.meta["external_id"],
                    "content_hash": data.content_hash,
                    "parent_ids": candidate.parent_ids,
                }
                for field in (
                    "embedding_profile",
                    "vector_space_id",
                    "collection_id",
                    "workflow_id",
                ):
                    if data.meta.get(field):
                        chunk_meta[field] = data.meta[field]
                if data.document_id:
                    try:
                        chunk_meta["document_id"] = str(uuid.UUID(str(data.document_id)))
                    except Exception:
                        chunk_meta["document_id"] = data.document_id
                results.append(
                    {
                        "content": chunk_text,
                        "normalized": normalized,
                        "meta": chunk_meta,
                    }
                )
                chunk_index += 1
        return results
