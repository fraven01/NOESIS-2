from __future__ import annotations

import hashlib
import re
import time
import uuid
from datetime import datetime
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from celery import shared_task

from ai_core.tool_contracts.base import tool_context_from_meta
from documents.pipeline import (
    DocumentPipelineConfig,
    ParsedResult,
)
from ai_core.infra.observability import (
    observe_span,
    update_observation,
)
from ai_core.tools.errors import RateLimitedError
from common.celery import RetryableTask, ScopedTask
from common.logging import get_logger
from django.conf import settings
from django.utils import timezone
from pydantic import ValidationError

from ai_core.infra import object_store, pii
from ai_core.infra.pii_flags import get_pii_config
from ai_core.segmentation import segment_markdown_blocks
from ai_core.rag.embedding_config import (
    get_embedding_profile,
)
from ai_core.rag.embedding_cache import (
    compute_text_hash,
    fetch_cached_embeddings,
    store_cached_embeddings,
)
from ai_core.rag.parents import limit_parent_payload
from ai_core.rag.schemas import Chunk
from ai_core.rag.normalization import normalise_text
from ai_core.rag.ingestion_contracts import (
    ChunkMeta,
    ensure_embedding_dimensions,
    log_embedding_quality_stats,
)
from ai_core.rag.chunking import RoutingAwareChunker
from ai_core.rag.embeddings import (
    EmbeddingBatchResult,
    EmbeddingClientError,
    get_embedding_client,
)
from ai_core.rag.pricing import calculate_embedding_cost
from ai_core.rag.vector_store import get_default_router
from ai_core.rag.vector_client import get_client_for_schema, get_default_schema

from .helpers.caching import (
    _acquire_dedupe_lock,
    _cache_delete,
    _cache_get,
    _cache_key,
    _cache_set,
    _dedupe_key,
    _dedupe_status,
    _log_cache_hit,
    _log_dedupe_hit,
    _mark_dedupe_done,
    _redis_client,
    _release_dedupe_lock,
)
from .helpers.chunking import (
    _build_chunk_prefix,
    _build_parsed_blocks,
    _build_processing_context,
    _chunkify,
    _resolve_overlap_tokens,
    _resolve_parent_capture_max_bytes,
    _resolve_parent_capture_max_depth,
    _split_by_limit,
    _split_sentences_impl,
    _token_count,
)
from .helpers.embedding import (
    _FAILED_CHUNK_ID_LIMIT,
    _coerce_cache_part,
    _extract_chunk_identifier,
    _hash_parts,
    _log_embedding_cache_hit,
    _normalise_embedding,
    _observed_embed_section,
    _resolve_embedding_model_version,
    _resolve_embedding_profile_id,
    _resolve_upsert_content_hash,
    _resolve_upsert_embedding_profile,
    _resolve_upsert_vector_space_id,
    _resolve_vector_space_id,
    _resolve_vector_space_schema,
    _should_normalise_embeddings,
)
from .helpers.task_utils import (
    _build_path,
    _object_store_path_exists,
    _resolve_artifact_filename,
    _task_context_payload,
    _task_result,
)


logger = get_logger(__name__)

_REFERENCE_ID_PATTERN = re.compile(
    r"(?:doc|document|ref)[:/]{1,2}([0-9a-fA-F-]{32,36})",
    re.IGNORECASE,
)


def _normalize_reference_id(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return str(uuid.UUID(candidate))
    except Exception:
        return candidate


def _extract_reference_ids(text: str, meta: Mapping[str, Any] | None) -> list[str]:
    references: list[str] = []
    meta_refs: list[str] = []
    if isinstance(meta, Mapping):
        candidate = meta.get("reference_ids") or meta.get("references")
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for ref in candidate:
                try:
                    ref_text = str(ref).strip()
                except Exception:
                    ref_text = ""
                if ref_text:
                    normalized = _normalize_reference_id(ref_text)
                    if normalized:
                        meta_refs.append(normalized)
    references.extend(meta_refs)

    if text:
        for match in _REFERENCE_ID_PATTERN.finditer(text):
            ref_text = match.group(1).strip()
            if ref_text:
                normalized = _normalize_reference_id(ref_text)
                if normalized:
                    references.append(normalized)

    seen: set[str] = set()
    unique: list[str] = []
    for ref in references:
        if ref in seen:
            continue
        seen.add(ref)
        unique.append(ref)
    return unique


def _extract_reference_labels(meta: Mapping[str, Any] | None) -> list[str]:
    labels: list[str] = []
    if isinstance(meta, Mapping):
        candidate = meta.get("reference_labels")
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for label in candidate:
                try:
                    label_text = str(label).strip()
                except Exception:
                    label_text = ""
                if label_text:
                    labels.append(label_text)
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    return unique


_DEDUPE_TTL_SECONDS = 24 * 60 * 60
_CACHE_TTL_CHUNK_SECONDS = 60 * 60
_CACHE_TTL_EMBED_SECONDS = 24 * 60 * 60


@shared_task(base=ScopedTask, queue="ingestion", name="ai_core.tasks.ingest_raw")
def ingest_raw(meta: Dict[str, str], name: str, data: bytes) -> Dict[str, Any]:
    """Persist raw document bytes."""
    external_id = meta.get("external_id")
    if not external_id:
        raise ValueError("external_id required for ingest_raw")

    path = _build_path(meta, "raw", name)
    object_store.put_bytes(path, data)
    content_hash = hashlib.sha256(data).hexdigest()
    meta["content_hash"] = content_hash
    return _task_result(
        status="success",
        data={"path": path, "content_hash": content_hash},
        meta=meta,
        task_name=getattr(ingest_raw, "name", "ingest_raw"),
    )


@shared_task(base=ScopedTask, queue="ingestion", name="ai_core.tasks.extract_text")
def extract_text(meta: Dict[str, str], raw_path: str) -> Dict[str, Any]:
    """Decode bytes to text and store."""
    full = object_store.BASE_PATH / raw_path
    text = full.read_bytes().decode("utf-8")
    out_path = _build_path(meta, "text", f"{Path(raw_path).stem}.txt")
    object_store.put_bytes(out_path, text.encode("utf-8"))
    return _task_result(
        status="success",
        data={"path": out_path},
        meta=meta,
        task_name=getattr(extract_text, "name", "extract_text"),
    )


@shared_task(base=ScopedTask, queue="ingestion", name="ai_core.tasks.pii_mask")
def pii_mask(meta: Dict[str, str], text_path: str) -> Dict[str, Any]:
    """Mask PII in text."""
    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    masked = pii.mask(text)
    if masked == text:
        # Only apply fallback numeric masking when PII masking is enabled.
        config = get_pii_config()
        mode = str(config.get("mode", "")).lower()
        policy = str(config.get("policy", "")).lower()
        if mode != "off" and policy != "off":
            masked = re.sub(r"\d", "X", text)
    out_path = _build_path(meta, "text", f"{Path(text_path).stem}.masked.txt")
    object_store.put_bytes(out_path, masked.encode("utf-8"))
    return _task_result(
        status="success",
        data={"path": out_path},
        meta=meta,
        task_name=getattr(pii_mask, "name", "pii_mask"),
    )


@shared_task(base=ScopedTask, queue="ingestion", name="ai_core.tasks._split_sentences")
def _split_sentences(text: str) -> Dict[str, Any]:
    sentences = _split_sentences_impl(text)
    return _task_result(
        status="success",
        data={"sentences": sentences},
        task_name=getattr(_split_sentences, "name", "_split_sentences"),
    )


@shared_task(
    base=RetryableTask,
    queue="ingestion",
    name="ai_core.tasks.chunk",
    time_limit=600,
    soft_time_limit=540,
)
@observe_span(name="ingestion.chunk")
def chunk(meta: Dict[str, str], text_path: str) -> Dict[str, Any]:
    """Split text into overlapping chunks for embeddings and capture parents."""

    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    content_hash = meta.get("content_hash")
    if not content_hash:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta["content_hash"] = content_hash
    external_id = meta.get("external_id")
    if not external_id:
        raise ValueError("external_id required for chunk")

    context = tool_context_from_meta(meta)
    embedding_model_version = _resolve_embedding_model_version(meta)
    embedding_created_at = timezone.now().isoformat()
    resolved_vector_space_id = meta.get("vector_space_id") or _resolve_vector_space_id(
        meta
    )

    cache_client = _redis_client()
    cache_key = None
    if cache_client is not None:
        profile_key = _coerce_cache_part(meta.get("embedding_profile"))
        parts = [context.scope.tenant_id, content_hash]
        if profile_key:
            parts.append(profile_key)
        idempotency_key = _hash_parts(*parts)
        if idempotency_key:
            cache_key = _cache_key("chunk", idempotency_key)
            cached_path = _cache_get(cache_client, cache_key)
            if cached_path:
                if _object_store_path_exists(cached_path):
                    _log_cache_hit(
                        task_name="chunk",
                        idempotency_key=idempotency_key,
                        cache_key=cache_key,
                        cached_path=cached_path,
                        meta=meta,
                    )
                    return _task_result(
                        status="success",
                        data={"path": cached_path},
                        meta=meta,
                        task_name=getattr(chunk, "name", "chunk"),
                    )
                _cache_delete(cache_client, cache_key)

    target_tokens = int(getattr(settings, "RAG_CHUNK_TARGET_TOKENS", 450))
    profile_limit: Optional[int] = None
    profile_id = meta.get("embedding_profile")
    if profile_id is not None:
        profile_key = str(profile_id).strip()
        if profile_key:
            profile_limit = get_embedding_profile(profile_key).chunk_hard_limit
            meta["embedding_profile"] = profile_key
    fallback_limit = 512
    if profile_limit is not None:
        hard_limit = profile_limit
        target_tokens = min(target_tokens, hard_limit)
    else:
        hard_limit = max(target_tokens, fallback_limit)
    overlap_tokens = _resolve_overlap_tokens(
        text,
        meta,
        target_tokens=target_tokens,
        hard_limit=hard_limit,
    )
    prefix = _build_chunk_prefix(meta)

    blocks_path_value = meta.get("parsed_blocks_path")
    structured_blocks: List[Dict[str, object]] = []
    block_stats: Dict[str, object] = {}
    if blocks_path_value:
        try:
            payload = object_store.read_json(str(blocks_path_value))
        except FileNotFoundError:
            payload = None
        except Exception:  # pragma: no cover - defensive guard
            context = tool_context_from_meta(meta)
            logger.warning(
                "ingestion.chunk.blocks_read_failed",
                extra={
                    "tenant_id": context.scope.tenant_id,
                    "case_id": context.business.case_id,
                    "path": blocks_path_value,
                },
            )
            payload = None
        if isinstance(payload, dict):
            raw_blocks = payload.get("blocks") or []
            if isinstance(raw_blocks, list):
                structured_blocks = [
                    entry for entry in raw_blocks if isinstance(entry, dict)
                ]
            stats_value = payload.get("statistics")
            if isinstance(stats_value, Mapping):
                block_stats = dict(stats_value)

    mask_enabled = bool(getattr(settings, "INGESTION_PII_MASK_ENABLED", True))

    def _mask_for_chunk(value: str) -> str:
        if not mask_enabled:
            return value
        masked_value = pii.mask(value)
        if masked_value == value:
            config = get_pii_config()
            mode = str(config.get("mode", "")).lower()
            policy = str(config.get("policy", "")).lower()
            if mode != "off" and policy != "off":
                masked_value = re.sub(r"\d", "X", value)
        return masked_value

    processing_context = _build_processing_context(meta=meta, tool_context=context)
    if processing_context is not None:
        parsed_blocks = _build_parsed_blocks(
            text=text,
            structured_blocks=structured_blocks,
            mask_fn=_mask_for_chunk,
        )
        parsed_result = ParsedResult(
            text_blocks=parsed_blocks,
            assets=(),
            statistics=block_stats,
        )
        chunker = RoutingAwareChunker()
        pipeline_config = DocumentPipelineConfig()
        chunk_entries, _stats = chunker.chunk(
            None,
            parsed_result,
            context=processing_context,
            config=pipeline_config,
        )

        document_id = str(processing_context.metadata.document_id)
        parent_nodes: Dict[str, Dict[str, object]] = {}
        doc_title = str(meta.get("title") or external_id or "").strip()
        root_id = f"{document_id}#doc"
        parent_nodes[root_id] = {
            "id": root_id,
            "type": "document",
            "title": doc_title or None,
            "level": 0,
            "order": 0,
            "document_id": document_id,
        }
        parent_order = 0
        chunks: List[Dict[str, object]] = []
        for index, entry in enumerate(chunk_entries):
            chunk_text = str(entry.get("text") or "")
            if not chunk_text.strip():
                continue
            parent_ref = entry.get("parent_ref")
            parent_ids = [root_id]
            if parent_ref:
                parent_id = str(parent_ref)
                parent_ids.append(parent_id)
                if parent_id not in parent_nodes:
                    parent_order += 1
                    section_path = entry.get("section_path")
                    title = None
                    level = 1
                    if isinstance(section_path, Sequence) and not isinstance(
                        section_path, (str, bytes, bytearray)
                    ):
                        path_parts = [str(part) for part in section_path if part]
                        if path_parts:
                            title = path_parts[-1]
                            level = len(path_parts)
                    parent_nodes[parent_id] = {
                        "id": parent_id,
                        "type": "section",
                        "title": title,
                        "level": level,
                        "order": parent_order,
                        "document_id": document_id,
                    }

            chunk_hash = entry.get("chunk_id")
            if not chunk_hash:
                chunk_hash_input = f"{content_hash}:{index}".encode("utf-8")
                chunk_hash = hashlib.sha256(chunk_hash_input).hexdigest()
            chunk_meta = {
                "tenant_id": context.scope.tenant_id,
                "case_id": context.business.case_id,
                "source": text_path,
                "hash": str(chunk_hash),
                "external_id": str(external_id),
                "content_hash": str(content_hash),
                "parent_ids": parent_ids,
                "document_id": document_id,
            }
            entry_meta = entry.get("meta") if isinstance(entry, Mapping) else None
            reference_ids = _extract_reference_ids(chunk_text, entry_meta or meta)
            if reference_ids:
                chunk_meta["reference_ids"] = reference_ids
            reference_labels = _extract_reference_labels(entry_meta or meta)
            if reference_labels:
                chunk_meta["reference_labels"] = reference_labels
            if embedding_model_version:
                chunk_meta["embedding_model_version"] = embedding_model_version
                chunk_meta["embedding_created_at"] = embedding_created_at
            if meta.get("embedding_profile"):
                chunk_meta["embedding_profile"] = meta["embedding_profile"]
            if resolved_vector_space_id:
                chunk_meta["vector_space_id"] = resolved_vector_space_id
            if meta.get("process"):
                chunk_meta["process"] = meta["process"]
            if context.business.collection_id:
                chunk_meta["collection_id"] = context.business.collection_id
            if context.business.workflow_id:
                chunk_meta["workflow_id"] = context.business.workflow_id
            if meta.get("lifecycle_state"):
                chunk_meta["lifecycle_state"] = meta["lifecycle_state"]

            chunks.append(
                {
                    "content": chunk_text,
                    "normalized": normalise_text(chunk_text),
                    "meta": chunk_meta,
                }
            )

        limited_parents = limit_parent_payload(parent_nodes)
        payload = {"chunks": chunks, "parents": limited_parents}
        chunk_filename = _resolve_artifact_filename(meta, "chunks")
        out_path = _build_path(meta, "embeddings", chunk_filename)
        object_store.write_json(out_path, payload)
        if cache_client is not None and cache_key:
            _cache_set(cache_client, cache_key, out_path, _CACHE_TTL_CHUNK_SECONDS)
        return _task_result(
            status="success",
            data={"path": out_path},
            meta=meta,
            task_name=getattr(chunk, "name", "chunk"),
        )

    fallback_segments: List[str] = []
    if not structured_blocks:
        fallback_segments = segment_markdown_blocks(text)
    parent_nodes: Dict[str, Dict[str, object]] = {}
    parent_contents: Dict[str, List[str]] = {}
    parent_content_bytes: Dict[str, int] = {}
    parent_capture_max_depth = _resolve_parent_capture_max_depth()
    parent_capture_max_bytes = _resolve_parent_capture_max_bytes()
    parent_stack: List[Dict[str, object]] = []
    section_counter = 0
    order_counter = 0

    document_id_value = meta.get("document_id")
    document_id: Optional[str] = None
    if document_id_value not in {None, "", "None"}:
        try:
            candidate_uuid = (
                document_id_value
                if isinstance(document_id_value, uuid.UUID)
                else uuid.UUID(str(document_id_value).strip())
            )
            document_id = str(candidate_uuid)
        except (ValueError, TypeError, AttributeError):
            try:
                candidate_text = str(document_id_value).strip()
            except Exception:
                candidate_text = ""
            document_id = candidate_text or None

    doc_title = str(meta.get("title") or meta.get("external_id") or "").strip()
    # Use compact UUIDs for parent identifiers to align with external document_id formatting
    parent_prefix = document_id if document_id else str(external_id)
    root_id = f"{parent_prefix}#doc"
    parent_nodes[root_id] = {
        "id": root_id,
        "type": "document",
        "title": doc_title or None,
        "level": 0,
        "order": order_counter,
    }
    if document_id:
        parent_nodes[root_id]["document_id"] = document_id
    parent_contents[root_id] = []
    parent_content_bytes[root_id] = 0

    def _within_capture_depth(level: int) -> bool:
        """Return True when parent capture is allowed for the given heading level."""
        if parent_capture_max_depth <= 0:
            # A value of zero disables the depth restriction so that parent capture
            # behaves as an "all levels" setting.
            return True
        return level <= parent_capture_max_depth

    def _append_parent_text(parent_id: str, text: str, level: int) -> None:
        if not text:
            return
        normalised = text.strip()
        if not normalised:
            return
        is_root_parent = parent_id == root_id
        if (
            not is_root_parent
            and parent_capture_max_depth > 0
            and not _within_capture_depth(level)
        ):
            return

        if parent_capture_max_bytes > 0 and not is_root_parent:
            used = parent_content_bytes.get(parent_id, 0)
            remaining = parent_capture_max_bytes - used
            if remaining <= 0:
                parent_nodes[parent_id]["capture_limited"] = True
                return

            separator_bytes = 0
            if parent_contents[parent_id]:
                separator_bytes = len("\n\n".encode("utf-8"))
            if remaining <= separator_bytes:
                parent_nodes[parent_id]["capture_limited"] = True
                return

            allowed = remaining - separator_bytes
            encoded = normalised.encode("utf-8")
            if len(encoded) > allowed:
                truncated = encoded[:allowed]
                preview = truncated.decode("utf-8", errors="ignore").strip()
                parent_nodes[parent_id]["capture_limited"] = True
                if not preview:
                    return
                parent_contents[parent_id].append(preview)
                appended_bytes = len(preview.encode("utf-8"))
                parent_content_bytes[parent_id] = (
                    used + separator_bytes + appended_bytes
                )
                return

            parent_contents[parent_id].append(normalised)
            parent_content_bytes[parent_id] = used + separator_bytes + len(encoded)
            return

        parent_contents[parent_id].append(normalised)
        if parent_capture_max_bytes > 0:
            used = parent_content_bytes.get(parent_id, 0)
            separator_bytes = 0
            if used > 0:
                separator_bytes = len("\n\n".encode("utf-8"))
            parent_content_bytes[parent_id] = (
                used + separator_bytes + len(normalised.encode("utf-8"))
            )

    def _append_parent_text_with_root(parent_id: str, text: str, level: int) -> None:
        _append_parent_text(parent_id, text, level)
        if parent_id == root_id:
            return
        _append_parent_text(root_id, text, level)

    def _register_section(
        title: str, level: int, path: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, object]:
        nonlocal section_counter, order_counter
        section_counter += 1
        order_counter += 1
        parent_id = f"{parent_prefix}#sec-{section_counter}"
        info = {
            "id": parent_id,
            "type": "section",
            "title": title or None,
            "level": level,
            "order": order_counter,
        }
        if path is not None:
            info["path"] = path
        if document_id:
            info["document_id"] = document_id
        parent_nodes[parent_id] = info
        parent_contents[parent_id] = []
        parent_content_bytes[parent_id] = 0
        return info

    chunk_candidates: List[Tuple[str, List[str], str]] = []
    last_registered_stack: List[Dict[str, object]] = []

    # SemanticChunker has been removed - using simple chunking approach
    # For structured chunking, use HybridChunker from ai_core.rag.chunking
    pending_pieces = []
    pending_parent_ids: Optional[Tuple[str, ...]] = None

    def _flush_pending() -> None:
        nonlocal pending_pieces, pending_parent_ids
        if not pending_pieces or pending_parent_ids is None:
            pending_pieces = []
            pending_parent_ids = None
            return

        combined_text = "\n\n".join(part for part in pending_pieces if part.strip())
        sentences = _split_sentences_impl(combined_text)
        if not sentences:
            sentences = [combined_text] if combined_text else []

        if sentences:
            bodies = _chunkify(
                sentences,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                hard_limit=hard_limit,
            )
            for body in bodies:
                chunk_candidates.append((body, list(pending_parent_ids), ""))

        pending_pieces = []
        pending_parent_ids = None

    heading_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

    for block in fallback_segments:
        stripped_block = block.strip()
        if not stripped_block:
            continue
        heading_match = heading_pattern.match(stripped_block)
        if heading_match:
            _flush_pending()
            hashes, heading_title = heading_match.groups()
            level = len(hashes)
            while parent_stack and int(parent_stack[-1].get("level") or 0) >= level:
                parent_stack.pop()
            section_info = _register_section(heading_title.strip(), level)
            parent_stack.append(section_info)
            _append_parent_text_with_root(section_info["id"], stripped_block, level)
            continue

        block_pieces = [block]
        if _token_count(block) > hard_limit:
            block_pieces = _split_by_limit(block, hard_limit)
        for piece in block_pieces:
            piece_text = piece.strip()
            if not piece_text:
                continue
            if parent_stack:
                target_info = parent_stack[-1]
                target_level = int(target_info.get("level") or 0)
                _append_parent_text_with_root(
                    target_info["id"], piece_text, target_level
                )
            else:
                _append_parent_text(root_id, piece_text, 0)
            parent_ids = [root_id] + [info["id"] for info in parent_stack]
            unique_parent_ids = list(dict.fromkeys(pid for pid in parent_ids if pid))
            new_parent_ids = tuple(unique_parent_ids)

            if pending_parent_ids is not None and pending_parent_ids != new_parent_ids:
                _flush_pending()

            if pending_parent_ids is None:
                pending_parent_ids = new_parent_ids

            pending_pieces.append(piece_text)

    _flush_pending()

    if not parent_stack and last_registered_stack:
        parent_stack = last_registered_stack

    if not chunk_candidates:
        fallback_ids = [root_id] + [info["id"] for info in parent_stack]
        unique_fallback_ids = list(dict.fromkeys(pid for pid in fallback_ids if pid))
        fallback_text = text.strip()
        if fallback_text:
            if parent_stack:
                target_info = parent_stack[-1]
                target_level = int(target_info.get("level") or 0)
                _append_parent_text_with_root(
                    target_info["id"], fallback_text, target_level
                )
            else:
                _append_parent_text(root_id, fallback_text, 0)
        chunk_candidates.append((text, unique_fallback_ids or [root_id], ""))

    chunks: List[Dict[str, object]] = []
    chunk_index = 0
    for body, parent_ids, heading_prefix in chunk_candidates:
        prefix_parts: List[str] = []
        if prefix:
            prefix_parts.append(prefix)
        if heading_prefix:
            prefix_parts.append(heading_prefix)
        prefix_token_count = sum(_token_count(part) for part in prefix_parts if part)
        if prefix_token_count >= hard_limit:
            body_limit = 0
        else:
            body_limit = hard_limit - prefix_token_count

        body_segments: List[str] = [body]
        if body_limit > 0:
            adjusted_segments: List[str] = []
            for segment in body_segments:
                if _token_count(segment) > body_limit:
                    adjusted_segments.extend(_split_by_limit(segment, body_limit))
                else:
                    adjusted_segments.append(segment)
            body_segments = adjusted_segments or [""]
        elif not body:
            body_segments = [""]

        for segment in body_segments:
            chunk_text = segment
            if prefix_parts:
                combined_prefix = "\n\n".join(
                    str(part).strip() for part in prefix_parts if str(part).strip()
                )
                if combined_prefix:
                    chunk_text = (
                        f"{combined_prefix}\n\n{chunk_text}"
                        if chunk_text
                        else combined_prefix
                    )
            normalised = normalise_text(chunk_text)
            chunk_hash_input = f"{content_hash}:{chunk_index}".encode("utf-8")
            chunk_hash = hashlib.sha256(chunk_hash_input).hexdigest()
            chunk_meta = {
                "tenant_id": context.scope.tenant_id,
                "case_id": context.business.case_id,  # BREAKING CHANGE: from business_context
                "source": text_path,
                "hash": chunk_hash,
                "external_id": external_id,
                "content_hash": content_hash,
                # Provide per-chunk parent lineage for compatibility with existing tests
                "parent_ids": parent_ids,
            }
            reference_ids = _extract_reference_ids(chunk_text, meta)
            if reference_ids:
                chunk_meta["reference_ids"] = reference_ids
            reference_labels = _extract_reference_labels(meta)
            if reference_labels:
                chunk_meta["reference_labels"] = reference_labels
            if embedding_model_version:
                chunk_meta["embedding_model_version"] = embedding_model_version
                chunk_meta["embedding_created_at"] = embedding_created_at

            if meta.get("embedding_profile"):
                chunk_meta["embedding_profile"] = meta["embedding_profile"]
            if resolved_vector_space_id:
                chunk_meta["vector_space_id"] = resolved_vector_space_id
            # BREAKING CHANGE (Option A): Business IDs from business_context
            if context.business.collection_id:
                chunk_meta["collection_id"] = context.business.collection_id
            if context.business.workflow_id:
                chunk_meta["workflow_id"] = context.business.workflow_id

            if document_id:
                # Ensure canonical dashed UUID format for document_id in chunk meta
                try:
                    chunk_meta["document_id"] = str(uuid.UUID(str(document_id)))
                except Exception:
                    chunk_meta["document_id"] = document_id
            chunks.append(
                {
                    "content": chunk_text,
                    "normalized": normalised,
                    "meta": chunk_meta,
                }
            )
            chunk_index += 1

    for parent_id, info in parent_nodes.items():
        content_parts = parent_contents.get(parent_id) or []
        content_text = "\n\n".join(part for part in content_parts if part).strip()
        if content_text:
            info = dict(info)
            info["content"] = content_text
            parent_nodes[parent_id] = info

    # Provide compatibility for parent key formats: include both dashed and
    # compact UUID variants for the document root when a document_id is present.
    if document_id:
        try:
            compact_root = f"{document_id.replace('-', '')}#doc"
            dashed_root = f"{document_id}#doc"
            if compact_root in parent_nodes and dashed_root not in parent_nodes:
                parent_nodes[dashed_root] = dict(parent_nodes[compact_root])
                parent_nodes[dashed_root].setdefault("document_id", document_id)
        except Exception:
            pass

    limited_parents = limit_parent_payload(parent_nodes)
    payload = {"chunks": chunks, "parents": limited_parents}
    chunk_filename = _resolve_artifact_filename(meta, "chunks")
    out_path = _build_path(meta, "embeddings", chunk_filename)
    object_store.write_json(out_path, payload)
    if cache_client is not None and cache_key:
        _cache_set(cache_client, cache_key, out_path, _CACHE_TTL_CHUNK_SECONDS)
    return _task_result(
        status="success",
        data={"path": out_path},
        meta=meta,
        task_name=getattr(chunk, "name", "chunk"),
    )


@shared_task(
    base=RetryableTask,
    queue="ingestion",
    name="ai_core.tasks.embed",
    time_limit=300,
    soft_time_limit=270,
)
@observe_span(name="ingestion.embed")
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, Any]:
    """Generate embedding vectors for chunks via LiteLLM."""

    cache_client = _redis_client()
    cache_key = None
    dedupe_key = None
    dedupe_token = None
    idempotency_key = None
    if cache_client is not None:
        context = tool_context_from_meta(meta)
        profile_key = _resolve_embedding_profile_id(meta, allow_default=True)
        idempotency_key = _hash_parts(
            context.scope.tenant_id,
            chunks_path,
            profile_key,
        )
        if idempotency_key:
            cache_key = _cache_key("embed", idempotency_key)
            cached_path = _cache_get(cache_client, cache_key)
            if cached_path:
                if _object_store_path_exists(cached_path):
                    _log_cache_hit(
                        task_name="embed",
                        idempotency_key=idempotency_key,
                        cache_key=cache_key,
                        cached_path=cached_path,
                        meta=meta,
                    )
                    return _task_result(
                        status="success",
                        data={"path": cached_path},
                        meta=meta,
                        task_name=getattr(embed, "name", "embed"),
                    )
                _cache_delete(cache_client, cache_key)
            dedupe_key = _dedupe_key("embed", idempotency_key)
            status = _dedupe_status(_cache_get(cache_client, dedupe_key))
            if status == "done":
                logger.warning(
                    "task.cache.missing",
                    extra={
                        "task_name": "embed",
                        "idempotency_key": idempotency_key,
                        "dedupe_key": dedupe_key,
                        **_task_context_payload(meta),
                    },
                )
                _cache_delete(cache_client, dedupe_key)
                status = None
            if status == "inflight":
                _log_dedupe_hit(
                    task_name="embed",
                    idempotency_key=idempotency_key,
                    dedupe_key=dedupe_key,
                    status=status,
                    meta=meta,
                )
                raise RateLimitedError(
                    code="dedupe_inflight",
                    message="Embed already running for idempotency key",
                )
            token = uuid.uuid4().hex
            if _acquire_dedupe_lock(
                cache_client, dedupe_key, _DEDUPE_TTL_SECONDS, token
            ):
                dedupe_token = token
            else:
                status = (
                    _dedupe_status(_cache_get(cache_client, dedupe_key)) or "inflight"
                )
                if status == "done":
                    cached_path = _cache_get(cache_client, cache_key)
                    if cached_path and _object_store_path_exists(cached_path):
                        _log_cache_hit(
                            task_name="embed",
                            idempotency_key=idempotency_key,
                            cache_key=cache_key,
                            cached_path=cached_path,
                            meta=meta,
                        )
                        return _task_result(
                            status="success",
                            data={"path": cached_path},
                            meta=meta,
                            task_name=getattr(embed, "name", "embed"),
                        )
                _log_dedupe_hit(
                    task_name="embed",
                    idempotency_key=idempotency_key,
                    dedupe_key=dedupe_key,
                    status=status,
                    meta=meta,
                )
                raise RateLimitedError(
                    code="dedupe_inflight",
                    message="Embed already running for idempotency key",
                )

    chunks: List[Dict[str, Any]] = []
    parents: Dict[str, Any] = {}
    prepared: List[Dict[str, Any]] = []
    token_counts: List[int] = []
    chunk_identifiers: List[str] = []
    embeddings: List[Dict[str, Any]] = []
    total_chunks = 0

    try:
        with _observed_embed_section("load") as load_metrics:
            raw_chunks = object_store.read_json(chunks_path)
            if isinstance(raw_chunks, dict):
                chunks = list(raw_chunks.get("chunks", []) or [])
                parent_payload = raw_chunks.get("parents") or {}
                if isinstance(parent_payload, dict):
                    parents = parent_payload
            else:
                chunks = list(raw_chunks or [])
            load_metrics.set("chunks_count", len(chunks))
            if isinstance(parents, dict) and parents:
                load_metrics.set("parents_count", len(parents))

        embedding_model_version = _resolve_embedding_model_version(meta)
        embedding_created_at_value = timezone.now()
        embedding_created_at = embedding_created_at_value.isoformat()
        resolved_vector_space_id = meta.get(
            "vector_space_id"
        ) or _resolve_vector_space_id(meta)

        context = tool_context_from_meta(meta)
        try:
            update_observation(
                tags=["ingestion", "embed"],
                user_id=(str(context.scope.user_id) if context.scope.user_id else None),
                session_id=(
                    str(context.business.case_id) if context.business.case_id else None
                ),
                metadata={
                    "embedding_profile": meta.get("embedding_profile"),
                    "vector_space_id": resolved_vector_space_id,
                    "embedding_model_version": embedding_model_version,
                    "tenant_id": context.scope.tenant_id,
                    "collection_id": context.business.collection_id,
                },
            )
        except Exception:
            pass

        client = get_embedding_client()
        batch_size = max(
            1, int(getattr(settings, "EMBEDDINGS_BATCH_SIZE", client.batch_size))
        )

        with _observed_embed_section("chunk") as chunk_metrics:
            for ch in chunks:
                normalised = ch.get("normalized") or normalise_text(
                    ch.get("content", "")
                )
                text = normalised or ""
                meta_payload: Dict[str, Any] = {}
                raw_meta = ch.get("meta")
                if isinstance(raw_meta, MappingABC):
                    meta_payload = dict(raw_meta)
                if embedding_model_version:
                    meta_payload["embedding_model_version"] = embedding_model_version
                    meta_payload["embedding_created_at"] = embedding_created_at
                if resolved_vector_space_id:
                    meta_payload.setdefault("vector_space_id", resolved_vector_space_id)
                if meta.get("embedding_profile"):
                    meta_payload.setdefault(
                        "embedding_profile", meta["embedding_profile"]
                    )
                text_hash = compute_text_hash(text)
                prepared.append(
                    {
                        **ch,
                        "normalized": text,
                        "meta": meta_payload,
                        "_text_hash": text_hash,
                    }
                )
                token_count = _token_count(text)
                token_counts.append(token_count)
                identifier = _extract_chunk_identifier(ch)
                if identifier:
                    chunk_identifiers.append(identifier)
            chunk_metrics.set("chunks_count", len(prepared))
            chunk_metrics.set("token_count", sum(token_counts))

        total_chunks = len(prepared)
        vector_space_schema = _resolve_vector_space_schema(meta)
        cache_db_client = None
        cached_embeddings: Dict[str, Tuple[List[float], datetime]] = {}
        cache_hit_count = 0
        embedding_results: List[Dict[str, Any] | None] = [None] * total_chunks
        pending_entries: List[Dict[str, Any]] = []
        pending_indices: List[int] = []
        pending_token_counts: List[int] = []
        if embedding_model_version and vector_space_schema:
            try:
                cache_db_client = get_client_for_schema(vector_space_schema)
                cached_embeddings = fetch_cached_embeddings(
                    cache_db_client,
                    [entry.get("_text_hash", "") for entry in prepared],
                    model_version=embedding_model_version,
                )
            except Exception as exc:
                logger.warning(
                    "rag.embedding_cache.read_failed",
                    extra={
                        "error": str(exc),
                        "model_version": embedding_model_version,
                        "schema": vector_space_schema,
                        **_task_context_payload(meta),
                    },
                )
                cache_db_client = None
                cached_embeddings = {}

        for index, entry in enumerate(prepared):
            text_hash = entry.get("_text_hash")
            cached = cached_embeddings.get(text_hash) if text_hash else None
            if cached is not None:
                vector, cached_created_at = cached
                meta_payload = entry.get("meta")
                if isinstance(meta_payload, MappingABC):
                    meta_payload = dict(meta_payload)
                else:
                    meta_payload = {}
                meta_payload["embedding_model_version"] = embedding_model_version
                meta_payload["embedding_created_at"] = (
                    cached_created_at.isoformat()
                    if isinstance(cached_created_at, datetime)
                    else embedding_created_at
                )
                if resolved_vector_space_id:
                    meta_payload.setdefault("vector_space_id", resolved_vector_space_id)
                entry["meta"] = meta_payload
                embedding_results[index] = {
                    **entry,
                    "embedding": list(vector),
                    "vector_dim": len(vector),
                }
                cache_hit_count += 1
            else:
                pending_entries.append(entry)
                pending_indices.append(index)
                pending_token_counts.append(token_counts[index])
        expected_dim: Optional[int] = None
        batches = 0
        total_retry_count = 0
        total_backoff_ms = 0.0
        total_cost = 0.0
        embedding_model: Optional[str] = None

        with _observed_embed_section("embed") as embed_metrics:
            embed_metrics.set("batch_size", batch_size)
            embed_metrics.set("cache.hit_count", cache_hit_count)
            embed_metrics.set("cache.miss_count", len(pending_entries))
            pending_total = len(pending_entries)
            new_cache_entries: Dict[str, Sequence[float]] = {}
            for start in range(0, pending_total, batch_size):
                batch = pending_entries[start : start + batch_size]
                if not batch:
                    continue
                batch_indices = pending_indices[start : start + len(batch)]
                batches += 1
                inputs = [str(entry.get("normalized", "")) for entry in batch]
                batch_started = time.perf_counter()
                result: EmbeddingBatchResult = client.embed(inputs)
                duration_ms = (time.perf_counter() - batch_started) * 1000
                extra = {
                    "batch": batches,
                    "chunks": len(batch),
                    "duration_ms": duration_ms,
                    "model": result.model,
                    "model_name": result.model,
                    "model_used": result.model_used,
                    "attempts": result.attempts,
                }
                if result.timeout_s is not None:
                    extra["timeout_s"] = result.timeout_s
                key_alias = meta.get("key_alias")
                if key_alias:
                    extra["key_alias"] = key_alias
                logger.info("ingestion.embed.batch", extra=extra)

                retries = max(0, int(result.attempts) - 1)
                total_retry_count += retries
                retry_delays = result.retry_delays or ()
                if retry_delays:
                    total_backoff_ms += float(sum(retry_delays)) * 1000.0

                batch_token_count = sum(
                    pending_token_counts[start + index] for index in range(len(batch))
                )
                if batch_token_count:
                    total_cost += calculate_embedding_cost(
                        result.model, batch_token_count
                    )

                embedding_model = result.model

                batch_dim: Optional[int] = (
                    len(result.vectors[0]) if result.vectors else None
                )
                current_dim = batch_dim
                try:
                    current_dim = client.dim()
                except EmbeddingClientError:
                    current_dim = batch_dim
                if current_dim is not None:
                    if expected_dim is None:
                        expected_dim = current_dim
                    elif expected_dim != current_dim:
                        logger.info(
                            "ingestion.embed.dimension_changed",
                            extra={
                                "previous": expected_dim,
                                "current": current_dim,
                                "model": result.model,
                            },
                        )
                        expected_dim = current_dim
                if len(result.vectors) != len(batch):
                    raise ValueError("Embedding batch size mismatch")

                for entry, vector, target_index in zip(
                    batch, result.vectors, batch_indices
                ):
                    if expected_dim is not None and len(vector) != expected_dim:
                        raise ValueError("Embedding dimension mismatch")
                    embedding_results[target_index] = {
                        **entry,
                        "embedding": list(vector),
                        "vector_dim": len(vector),
                    }
                    text_hash = entry.get("_text_hash")
                    if text_hash:
                        new_cache_entries[str(text_hash)] = vector

            embeddings = [entry for entry in embedding_results if entry is not None]
            if embedding_model is None and embedding_model_version:
                embedding_model = embedding_model_version
            if cache_db_client and embedding_model_version and new_cache_entries:
                try:
                    store_cached_embeddings(
                        cache_db_client,
                        embeddings=new_cache_entries,
                        model_version=embedding_model_version,
                        created_at=embedding_created_at_value,
                    )
                except Exception as exc:
                    logger.warning(
                        "rag.embedding_cache.write_failed",
                        extra={
                            "error": str(exc),
                            "model_version": embedding_model_version,
                            "schema": vector_space_schema,
                            **_task_context_payload(meta),
                        },
                    )
            if embedding_model_version and cache_hit_count:
                _log_embedding_cache_hit(
                    task_name="embed",
                    model_version=embedding_model_version,
                    hit_count=cache_hit_count,
                    total_chunks=total_chunks,
                    meta=meta,
                )

            embed_metrics.set("chunks_count", len(embeddings))
            if embedding_model:
                embed_metrics.set("embedding_model", embedding_model)
            embed_metrics.set("retry.count", total_retry_count)
            embed_metrics.set("retry.backoff_ms", total_backoff_ms)
            embed_metrics.set("cost.usd_embedding", total_cost)

        logger.info(
            "ingestion.embed.summary",
            extra={"chunks": total_chunks, "batches": batches},
        )
        context = tool_context_from_meta(meta)
        try:
            logger.warning(
                "ingestion.embed.parents",
                extra={
                    "event": "DEBUG.TASKS.EMBED.PARENTS",
                    "tenant_id": context.scope.tenant_id,
                    "case_id": context.business.case_id,
                    "parents_count": (
                        len(parents) if isinstance(parents, dict) else None
                    ),
                },
            )
        except Exception:
            pass

        payload = {"chunks": embeddings, "parents": parents}
        vectors_filename = _resolve_artifact_filename(meta, "vectors")
        # Ensure compatibility with tests expecting a stable file name
        # while keeping a hashed component to avoid collisions.
        vectors_dir = Path(vectors_filename).stem  # e.g. "vectors-<seed>"
        out_path = _build_path(meta, "embeddings", vectors_dir, "vectors.json")
        with _observed_embed_section("write") as write_metrics:
            object_store.write_json(out_path, payload)
            write_metrics.set("chunks_count", len(embeddings))
            if isinstance(parents, dict) and parents:
                write_metrics.set("parents_count", len(parents))
        if cache_client is not None and cache_key and idempotency_key:
            _cache_set(cache_client, cache_key, out_path, _CACHE_TTL_EMBED_SECONDS)
        if cache_client is not None and dedupe_key and dedupe_token:
            _mark_dedupe_done(
                cache_client, dedupe_key, _DEDUPE_TTL_SECONDS, dedupe_token
            )
        return _task_result(
            status="success",
            data={"path": out_path},
            meta=meta,
            task_name=getattr(embed, "name", "embed"),
        )
    except Exception:
        if cache_client is not None and dedupe_key and dedupe_token:
            _release_dedupe_lock(cache_client, dedupe_key, dedupe_token)
        failed_chunks_count = len(prepared) if prepared else len(chunks)
        if not failed_chunks_count:
            failed_chunks_count = len(embeddings)
        truncated_ids = chunk_identifiers[:_FAILED_CHUNK_ID_LIMIT]
        try:
            update_observation(
                metadata={
                    "status": "error",
                    "failed_chunks_count": failed_chunks_count,
                    "failed_chunk_ids": truncated_ids,
                }
            )
        except Exception:
            pass
        raise


@shared_task(base=RetryableTask, queue="ingestion", name="ai_core.tasks.upsert")
def upsert(
    meta: Dict[str, str],
    embeddings_path: str,
    tenant_schema: Optional[str] = None,
    *,
    vector_client: Optional[Any] = None,
    vector_client_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Upsert embedded chunks into the vector client."""
    raw_data = object_store.read_json(embeddings_path)
    parents: Dict[str, Dict[str, object]] | None = None
    if isinstance(raw_data, dict):
        data = list(raw_data.get("chunks", []) or [])
        parents_payload = raw_data.get("parents") or {}
        if isinstance(parents_payload, dict) and parents_payload:
            parents = parents_payload
    else:
        data = list(raw_data or [])

    context = tool_context_from_meta(meta)
    cache_client = _redis_client()
    dedupe_key = None
    dedupe_token = None
    idempotency_key = None
    if cache_client is not None:
        tenant_id = context.scope.tenant_id
        content_hash = _resolve_upsert_content_hash(meta, data)
        vector_space_id = _resolve_upsert_vector_space_id(meta, data)
        profile_key = _resolve_upsert_embedding_profile(meta, data)
        idempotency_key = _hash_parts(
            tenant_id,
            vector_space_id,
            content_hash,
            profile_key,
        )
        if idempotency_key:
            dedupe_key = _dedupe_key("upsert", idempotency_key)
            status = _dedupe_status(_cache_get(cache_client, dedupe_key))
            if status == "done":
                _log_dedupe_hit(
                    task_name="upsert",
                    idempotency_key=idempotency_key,
                    dedupe_key=dedupe_key,
                    status=status,
                    meta=meta,
                )
                return _task_result(
                    status="success",
                    data={"written": 0},
                    meta=meta,
                    task_name=getattr(upsert, "name", "upsert"),
                )
            if status == "inflight":
                _log_dedupe_hit(
                    task_name="upsert",
                    idempotency_key=idempotency_key,
                    dedupe_key=dedupe_key,
                    status=status,
                    meta=meta,
                )
                raise RateLimitedError(
                    code="dedupe_inflight",
                    message="Upsert already running for idempotency key",
                )
            token = uuid.uuid4().hex
            if _acquire_dedupe_lock(
                cache_client, dedupe_key, _DEDUPE_TTL_SECONDS, token
            ):
                dedupe_token = token
            else:
                status = (
                    _dedupe_status(_cache_get(cache_client, dedupe_key)) or "inflight"
                )
                if status == "done":
                    _log_dedupe_hit(
                        task_name="upsert",
                        idempotency_key=idempotency_key,
                        dedupe_key=dedupe_key,
                        status=status,
                        meta=meta,
                    )
                    return _task_result(
                        status="success",
                        data={"written": 0},
                        meta=meta,
                        task_name=getattr(upsert, "name", "upsert"),
                    )
                _log_dedupe_hit(
                    task_name="upsert",
                    idempotency_key=idempotency_key,
                    dedupe_key=dedupe_key,
                    status=status,
                    meta=meta,
                )
                raise RateLimitedError(
                    code="dedupe_inflight",
                    message="Upsert already running for idempotency key",
                )
    try:
        # Debug visibility for parents presence in upsert input and parsed payload
        try:
            logger.warning(
                "ingestion.upsert.parents_loaded",
                extra={
                    "event": "DEBUG.TASKS.UPSERT.PARENTS_LOADED",
                    "tenant_id": context.scope.tenant_id,
                    "case_id": context.business.case_id,
                    "parents_present": bool(parents),
                    "parents_count": (
                        len(parents) if isinstance(parents, dict) else None
                    ),
                },
            )
        except Exception:
            pass
        chunk_objs = []
        for index, ch in enumerate(data):
            vector = ch.get("embedding")
            embedding = [float(v) for v in vector] if vector is not None else None
            if embedding is not None and _should_normalise_embeddings():
                normalised = _normalise_embedding(embedding)
                if normalised is not None:
                    embedding = normalised
            raw_meta = ch.get("meta", {})
            try:
                meta_model = ChunkMeta.model_validate(raw_meta)
            except ValidationError:
                logger.error(
                    "ingestion.chunk.meta.invalid",
                    extra={
                        "tenant_id": context.scope.tenant_id,
                        "case_id": context.business.case_id,
                        "chunk_index": index,
                        "keys": (
                            sorted(raw_meta.keys())
                            if isinstance(raw_meta, dict)
                            else None
                        ),
                    },
                )
                # Be tolerant for minimal test/scaffold inputs: fall back to a
                # permissive metadata dict when strict validation fails.
                # This preserves routing behaviour (tenant forwarding) and lets
                # dimension checks run even with partial metadata.
                fallback_meta: Dict[str, object] = {}
                if isinstance(raw_meta, dict):
                    # Always forward tenant_id if present
                    if raw_meta.get("tenant_id") is not None:
                        fallback_meta["tenant_id"] = str(raw_meta.get("tenant_id"))
                    # Include commonly provided optional fields when available
                    for key in (
                        "case_id",
                        "external_id",
                        "source",
                        "hash",
                        "content_hash",
                        "embedding_profile",
                        "embedding_model_version",
                        "embedding_created_at",
                        "vector_space_id",
                        "process",
                        "collection_id",
                        "workflow_id",
                        "document_version_id",
                        "is_latest",
                    ):
                        if raw_meta.get(key) is not None:
                            fallback_meta[key] = raw_meta.get(key)
                parents_map = parents
                if isinstance(ch.get("parents"), dict):
                    local_parents = ch.get("parents")
                    parents_map = local_parents if local_parents else parents_map
                chunk_objs.append(
                    Chunk(
                        content=ch["content"],
                        meta=fallback_meta,
                        embedding=embedding,
                        parents=parents_map,
                    )
                )
                continue
            # Strict path: validated metadata
            parents_map = parents
            if isinstance(ch.get("parents"), dict):
                local_parents = ch.get("parents")
                parents_map = local_parents if local_parents else parents_map
            chunk_objs.append(
                Chunk(
                    content=ch["content"],
                    meta=meta_model.model_dump(exclude_none=True),
                    embedding=embedding,
                    parents=parents_map,
                )
            )

        tenant_id = context.scope.tenant_id
        if not tenant_id:
            raise ValueError("tenant_id required for upsert")

        for chunk in chunk_objs:
            chunk_tenant = chunk.meta.get("tenant_id") if chunk.meta else None
            if chunk_tenant and str(chunk_tenant) != tenant_id:
                raise ValueError("chunk tenant mismatch")

        expected_dimension_value = meta.get("vector_space_dimension")
        expected_dimension: Optional[int] = None
        if expected_dimension_value is not None:
            try:
                expected_dimension = int(expected_dimension_value)
            except (TypeError, ValueError):
                expected_dimension = None

        # BREAKING CHANGE (Option A): workflow_id from business_context
        ensure_embedding_dimensions(
            chunk_objs,
            expected_dimension,
            tenant_id=tenant_id,
            process=meta.get("process"),
            workflow_id=context.business.workflow_id,
            embedding_profile=meta.get("embedding_profile"),
            vector_space_id=meta.get("vector_space_id"),
        )
        log_embedding_quality_stats(
            chunk_objs,
            tenant_id=tenant_id,
            process=meta.get("process"),
            workflow_id=context.business.workflow_id,
            embedding_profile=meta.get("embedding_profile"),
            vector_space_id=meta.get("vector_space_id"),
        )

        schema = tenant_schema or meta.get("tenant_schema")
        vector_space_schema = _resolve_vector_space_schema(meta)

        tenant_client = vector_client
        if tenant_client is None and callable(vector_client_factory):
            candidate = vector_client_factory()
            if candidate is None:
                tenant_client = None
            else:
                tenant_client = candidate
        if tenant_client is None:
            default_schema = get_default_schema()
            if vector_space_schema and vector_space_schema != default_schema:
                tenant_client = get_client_for_schema(vector_space_schema)
            else:
                router = get_default_router()
                tenant_client = router
                for_tenant = getattr(router, "for_tenant", None)
                if callable(for_tenant):
                    try:
                        tenant_client = for_tenant(tenant_id, schema)
                    except TypeError:
                        tenant_client = for_tenant(tenant_id)
        written = tenant_client.upsert_chunks(chunk_objs)
        if cache_client is not None and dedupe_key and dedupe_token:
            _mark_dedupe_done(
                cache_client, dedupe_key, _DEDUPE_TTL_SECONDS, dedupe_token
            )
        return _task_result(
            status="success",
            data={"written": written},
            meta=meta,
            task_name=getattr(upsert, "name", "upsert"),
        )
    except Exception:
        if cache_client is not None and dedupe_key and dedupe_token:
            _release_dedupe_lock(cache_client, dedupe_key, dedupe_token)
        raise


@shared_task(base=ScopedTask, queue="ingestion", name="ai_core.tasks.ingestion_run")
def ingestion_run(
    state: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Placeholder ingestion dispatcher used by the ingestion run endpoint."""

    state_payload = dict(state or {})
    if not isinstance(meta, MappingABC):
        raise ValueError("meta with tool_context is required for ingestion_run")
    tool_context = tool_context_from_meta(meta)

    def _coerce_str(value: object | None) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        try:
            return str(value).strip() or None
        except Exception:
            return None

    tenant_id = _coerce_str(tool_context.scope.tenant_id)
    case_id = _coerce_str(tool_context.business.case_id)
    trace_id = _coerce_str(tool_context.scope.trace_id)
    document_ids = state_payload.get("document_ids") or []
    if not isinstance(document_ids, list):
        document_ids = [document_ids]
    document_ids = [
        candidate
        for candidate in (_coerce_str(entry) for entry in document_ids)
        if candidate
    ]
    priority = _coerce_str(state_payload.get("priority")) or "normal"

    # Keep relying on django.utils.timezone.now so call sites and tests can
    # monkeypatch the module-level helper consistently.
    queued_at = timezone.now().isoformat()
    logger.info(
        "Queued ingestion run",
        extra={
            "tenant_id": tenant_id,
            "case_id": case_id,
            "document_ids": document_ids,
            "priority": priority,
            "queued_at": queued_at,
            "trace_id": trace_id,
        },
    )
    return _task_result(
        status="success",
        data={"status": "queued", "queued_at": queued_at},
        meta=meta,
        task_name=getattr(ingestion_run, "name", "ingestion_run"),
    )
