from __future__ import annotations

import hashlib
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

from celery import shared_task
from ai_core.infra.observability import observe_span, record_span, update_observation
from common.celery import ScopedTask
from common.logging import get_logger
from django.conf import settings
from django.utils import timezone
from pydantic import ValidationError

from .infra import object_store, pii
from .infra.pii_flags import get_pii_config
from .segmentation import segment_markdown_blocks
from .rag import metrics
from .rag.embedding_config import get_embedding_profile
from .rag.parents import limit_parent_payload
from .rag.schemas import Chunk
from .rag.normalization import normalise_text
from .rag.ingestion_contracts import ChunkMeta, ensure_embedding_dimensions
from .rag.embeddings import (
    EmbeddingBatchResult,
    EmbeddingClientError,
    get_embedding_client,
)
from .rag.vector_store import get_default_router

_ZERO_EPSILON = 1e-12


def _should_normalise_embeddings() -> bool:
    env_value = os.getenv("RAG_NEAR_DUPLICATE_REQUIRE_UNIT_NORM")
    if env_value is not None:
        lowered = env_value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    try:
        return bool(getattr(settings, "RAG_NEAR_DUPLICATE_REQUIRE_UNIT_NORM"))
    except Exception:
        return False


def _normalise_embedding(values: Sequence[float] | None) -> List[float] | None:
    if not values:
        return None
    try:
        floats = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    norm_sq = math.fsum(value * value for value in floats)
    if norm_sq <= _ZERO_EPSILON:
        return None
    norm = math.sqrt(norm_sq)
    if not math.isfinite(norm) or norm <= _ZERO_EPSILON:
        return None
    scale = 1.0 / norm
    return [value * scale for value in floats]


logger = get_logger(__name__)


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    tenant = object_store.sanitize_identifier(meta["tenant_id"])
    case = object_store.sanitize_identifier(meta["case_id"])
    return "/".join([tenant, case, *parts])


def log_ingestion_run_start(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
) -> None:
    extra = {
        "tenant_id": tenant,
        "case_id": case,
        "run_id": run_id,
        "doc_count": doc_count,
    }
    if trace_id:
        extra["trace_id"] = trace_id
    if idempotency_key:
        extra["idempotency_key"] = idempotency_key
    if embedding_profile:
        extra["embedding_profile"] = embedding_profile
    if vector_space_id:
        extra["vector_space_id"] = vector_space_id
    logger.info("ingestion.start", extra=extra)
    if trace_id:
        record_span(
            "rag.ingestion.run.start",
            trace_id=trace_id,
            attributes={**extra},
        )


def log_ingestion_run_end(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    inserted: int,
    replaced: int,
    skipped: int,
    total_chunks: int,
    duration_ms: float,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
) -> None:
    extra = {
        "tenant_id": tenant,
        "case_id": case,
        "run_id": run_id,
        "doc_count": doc_count,
        "inserted": inserted,
        "replaced": replaced,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }
    if trace_id:
        extra["trace_id"] = trace_id
    if idempotency_key:
        extra["idempotency_key"] = idempotency_key
    if embedding_profile:
        extra["embedding_profile"] = embedding_profile
    if vector_space_id:
        extra["vector_space_id"] = vector_space_id
    logger.info("ingestion.end", extra=extra)
    metrics.INGESTION_RUN_MS.observe(float(duration_ms))
    if trace_id:
        record_span(
            "rag.ingestion.run.end",
            trace_id=trace_id,
            attributes={**extra},
        )


@shared_task(base=ScopedTask, accepts_scope=True)
def ingest_raw(meta: Dict[str, str], name: str, data: bytes) -> Dict[str, str]:
    """Persist raw document bytes."""
    external_id = meta.get("external_id")
    if not external_id:
        raise ValueError("external_id required for ingest_raw")

    path = _build_path(meta, "raw", name)
    object_store.put_bytes(path, data)
    content_hash = hashlib.sha256(data).hexdigest()
    meta["content_hash"] = content_hash
    return {"path": path, "content_hash": content_hash}


@shared_task(base=ScopedTask, accepts_scope=True)
def extract_text(meta: Dict[str, str], raw_path: str) -> Dict[str, str]:
    """Decode bytes to text and store."""
    full = object_store.BASE_PATH / raw_path
    text = full.read_bytes().decode("utf-8")
    out_path = _build_path(meta, "text", f"{Path(raw_path).stem}.txt")
    object_store.put_bytes(out_path, text.encode("utf-8"))
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
def pii_mask(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
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
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
def _split_sentences(text: str) -> List[str]:
    """Best-effort sentence segmentation that retains punctuation."""

    pattern = re.compile(r"[^.!?]+(?:[.!?]+|\Z)")
    sentences = [segment.strip() for segment in pattern.findall(text)]
    sentences = [s for s in sentences if s]
    if sentences:
        if len(sentences) == 1 and not re.search(r"[.!?]", text):
            sentences = []
        else:
            return sentences
    # Fallback: use paragraphs or lines if no sentence boundary detected
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    return parts or [text.strip()]


if tiktoken:
    try:
        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - defensive fallback
        _TOKEN_ENCODING = None
else:
    _TOKEN_ENCODING = None


_FORCE_WHITESPACE_TOKENIZER = os.getenv(
    "AI_CORE_FORCE_WHITESPACE_TOKENIZER", ""
).lower() in (
    "1",
    "true",
    "yes",
)

_PRONOUN_PATTERN = re.compile(
    r"\b(ich|mich|mir|du|dich|dir|er|ihn|ihm|sie|ihr|es|wir|uns|euch|"
    r"they|them|their|theirs|he|him|his|she|her|hers|we|us|our|ours|i|me|my|mine)\b",
    re.IGNORECASE,
)

_LIST_LIKE_KEYWORDS = (
    "faq",
    "liste",
    "list",
    "bullet",
    "checklist",
    "glossary",
    "table",
)

_NARRATIVE_KEYWORDS = (
    "narrative",
    "narrativ",
    "story",
    "bericht",
    "report",
    "fallstudie",
    "memo",
    "conversation",
    "transkript",
)


def _should_use_tiktoken() -> bool:
    return _TOKEN_ENCODING is not None and not _FORCE_WHITESPACE_TOKENIZER


def set_tokenizer_override(force_whitespace: bool) -> None:
    global _FORCE_WHITESPACE_TOKENIZER
    _FORCE_WHITESPACE_TOKENIZER = force_whitespace


@contextmanager
def force_whitespace_tokenizer() -> Iterator[None]:
    """Temporarily force the whitespace-based tokenizer fallback."""

    previous = _FORCE_WHITESPACE_TOKENIZER
    set_tokenizer_override(True)
    try:
        yield
    finally:
        set_tokenizer_override(previous)


def _token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    if _should_use_tiktoken():
        # `disallowed_special=()` ensures consistent behaviour across tiktoken versions.
        return max(1, len(_TOKEN_ENCODING.encode(stripped, disallowed_special=())))

    whitespace_tokens = [segment for segment in stripped.split() if segment]
    if len(whitespace_tokens) > 1:
        return len(whitespace_tokens)

    return max(1, len(stripped))


def _split_by_limit(text: str, hard_limit: int) -> List[str]:
    if not text:
        return []

    limit = max(1, int(hard_limit))

    if _should_use_tiktoken():
        token_ids = _TOKEN_ENCODING.encode(text, disallowed_special=())
        if not token_ids:
            return []
        if len(token_ids) <= limit:
            return [text]
        parts: List[str] = []
        for start in range(0, len(token_ids), limit):
            chunk_ids = token_ids[start : start + limit]
            parts.append(_TOKEN_ENCODING.decode(chunk_ids))
        return parts

    whitespace_chunks = list(re.finditer(r"\S+\s*", text))
    if len(whitespace_chunks) > 1:
        if len(whitespace_chunks) <= limit:
            return [text]
        parts: List[str] = []
        current_segments: List[str] = []
        current_tokens = 0

        for match in whitespace_chunks:
            segment = match.group(0)
            stripped = segment.strip()

            if not stripped:
                continue

            if current_tokens + 1 > limit and current_segments:
                parts.append("".join(current_segments).rstrip())
                current_segments = []
                current_tokens = 0

            current_segments.append(segment)
            current_tokens += 1

        if current_segments:
            parts.append("".join(current_segments).rstrip())

        return [part for part in parts if part]

    if len(text) <= limit:
        return [text]

    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _estimate_overlap_ratio(text: str, meta: Dict[str, str]) -> float:
    """Estimate chunk overlap ratio between 10% and 25%."""

    ratio_min = 0.10
    ratio_max = 0.25
    stripped = text.strip()
    if not stripped:
        return ratio_min

    ratio = 0.15
    doc_type = str(
        meta.get("doc_class")
        or meta.get("collection_id")
        or meta.get("document_type")
        or meta.get("type")
        or ""
    ).lower()
    if doc_type:
        if any(keyword in doc_type for keyword in _LIST_LIKE_KEYWORDS):
            return ratio_min
        if any(keyword in doc_type for keyword in _NARRATIVE_KEYWORDS):
            ratio = max(ratio, 0.22)

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        bullet_lines = sum(
            1 for line in lines if re.match(r"^(?:[-*•]\s|\d+\.\s)", line)
        )
        bullet_ratio = bullet_lines / max(1, len(lines))
        if bullet_ratio >= 0.4:
            ratio -= 0.03
        elif bullet_ratio <= 0.1 and len(lines) > 4:
            ratio += 0.02

    words = re.findall(r"\b\w+\b", stripped)
    word_count = len(words)
    if word_count:
        pronoun_count = len(_PRONOUN_PATTERN.findall(stripped))
        pronoun_ratio = pronoun_count / word_count
        if pronoun_ratio >= 0.07:
            ratio += 0.07
        elif pronoun_ratio >= 0.04:
            ratio += 0.04
        elif pronoun_ratio <= 0.015:
            ratio -= 0.02

    return max(ratio_min, min(ratio, ratio_max))


def _resolve_overlap_tokens(
    text: str,
    meta: Dict[str, str],
    *,
    target_tokens: int,
    hard_limit: int,
) -> int:
    configured_limit = getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", None)
    try:
        configured_value = (
            int(configured_limit) if configured_limit is not None else None
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive
        configured_value = None

    if configured_value is not None and configured_value <= 0:
        return 0

    ratio = _estimate_overlap_ratio(text, meta)
    overlap = int(round(target_tokens * ratio))
    if ratio > 0 and overlap == 0:
        overlap = 1

    if configured_value is not None:
        overlap = min(overlap, configured_value)

    overlap = min(overlap, max(0, target_tokens - 1))

    return max(0, min(overlap, hard_limit))


def _chunkify(
    sentences: Sequence[str],
    *,
    target_tokens: int,
    overlap_tokens: int,
    hard_limit: int,
) -> List[str]:
    chunks: List[str] = []
    current: List[Tuple[str, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunk_text = " ".join(sentence for sentence, _ in current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if overlap_tokens > 0:
            retained: List[Tuple[str, int]] = []
            retained_tokens = 0
            for sentence, tokens in reversed(current):
                retained.insert(0, (sentence, tokens))
                retained_tokens += tokens
                if retained_tokens >= overlap_tokens:
                    break
            current = retained
            current_tokens = retained_tokens
        else:
            current = []
            current_tokens = 0

    for sentence in sentences:
        tokens = _token_count(sentence)
        if tokens > hard_limit:
            sub_sentences = _split_by_limit(sentence, hard_limit)
            for sub_sentence in sub_sentences:
                sub_tokens = _token_count(sub_sentence)
                if sub_tokens > hard_limit:
                    # Guard against pathological tokenizer fallbacks by forcing a hard trim.
                    sub_sentence = sub_sentence[:hard_limit]
                    sub_tokens = _token_count(sub_sentence)
                if current_tokens + sub_tokens > hard_limit:
                    flush()
                current.append((sub_sentence, sub_tokens))
                current_tokens += sub_tokens
                if current_tokens >= target_tokens:
                    flush()
            continue

        if current_tokens + tokens > hard_limit and current:
            flush()

        current.append((sentence, tokens))
        current_tokens += tokens

        if current_tokens >= target_tokens:
            flush()

    flush()
    return chunks


def _build_chunk_prefix(meta: Dict[str, str]) -> str:
    parts: List[str] = []
    breadcrumbs = meta.get("breadcrumbs")
    if isinstance(breadcrumbs, Iterable) and not isinstance(breadcrumbs, (str, bytes)):
        crumb_parts = [str(item).strip() for item in breadcrumbs if str(item).strip()]
        if crumb_parts:
            parts.append(" / ".join(crumb_parts))
    title = meta.get("title")
    if title:
        parts.append(str(title).strip())
    return " — ".join(part for part in parts if part)


def _resolve_parent_capture_max_depth() -> int:
    try:
        value = getattr(settings, "RAG_PARENT_CAPTURE_MAX_DEPTH", 0)
    except Exception:
        return 0

    try:
        depth = int(value)
    except (TypeError, ValueError):
        return 0

    return depth if depth > 0 else 0


def _resolve_parent_capture_max_bytes() -> int:
    try:
        value = getattr(settings, "RAG_PARENT_CAPTURE_MAX_BYTES", 0)
    except Exception:
        return 0

    try:
        byte_limit = int(value)
    except (TypeError, ValueError):
        return 0

    return byte_limit if byte_limit > 0 else 0


def chunk(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
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
    if blocks_path_value:
        try:
            payload = object_store.read_json(str(blocks_path_value))
        except FileNotFoundError:
            payload = None
        except Exception:  # pragma: no cover - defensive guard
            logger.warning(
                "ingestion.chunk.blocks_read_failed",
                extra={
                    "tenant_id": meta.get("tenant_id"),
                    "case_id": meta.get("case_id"),
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

    document_id_value = meta.get("document_id") or meta.get("doc_id")
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
    parent_prefix = document_id or str(external_id)
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

    if structured_blocks:
        section_registry: Dict[Tuple[str, ...], Dict[str, object]] = {}

        for block in structured_blocks:
            text_value = str(block.get("text") or "")
            kind = str(block.get("kind") or "").lower()
            section_path_raw = block.get("section_path")
            path_tuple: Tuple[str, ...] = ()
            if isinstance(section_path_raw, (list, tuple)):
                path_tuple = tuple(
                    str(part).strip()
                    for part in section_path_raw
                    if isinstance(part, str) and str(part).strip()
                )
            masked_text = _mask_for_chunk(text_value)
            if kind == "heading":
                level = len(path_tuple) if path_tuple else 1
                while parent_stack and len(parent_stack[-1].get("path", ())) >= level:
                    parent_stack.pop()
                title_candidate = path_tuple[-1] if path_tuple else text_value.strip()
                section_info = _register_section(
                    title_candidate or text_value.strip(), level, path_tuple or None
                )
                section_registry[path_tuple] = section_info
                parent_stack.append(section_info)
                if masked_text.strip():
                    _append_parent_text_with_root(
                        section_info["id"], masked_text.strip(), level
                    )
                continue

            normalised_text = masked_text.strip()
            if not normalised_text:
                continue

            if path_tuple:
                desired_stack: List[Dict[str, object]] = []
                for depth in range(1, len(path_tuple) + 1):
                    prefix_tuple = path_tuple[:depth]
                    info = section_registry.get(prefix_tuple)
                    if info is None:
                        title_candidate = prefix_tuple[-1] if prefix_tuple else ""
                        info = _register_section(title_candidate, depth, prefix_tuple)
                        section_registry[prefix_tuple] = info
                    desired_stack.append(info)
                parent_stack = desired_stack
            else:
                parent_stack = []

            block_pieces = [normalised_text]
            if _token_count(normalised_text) > hard_limit:
                block_pieces = _split_by_limit(normalised_text, hard_limit)

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
                sentences = _split_sentences(piece_text)
                if not sentences:
                    continue
                bodies = _chunkify(
                    sentences,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    hard_limit=hard_limit,
                )
                if not bodies:
                    continue
                parent_ids = [root_id] + [info["id"] for info in parent_stack]
                unique_parent_ids = list(
                    dict.fromkeys(pid for pid in parent_ids if pid)
                )
                heading_prefix = " / ".join(path_tuple) if path_tuple else ""
                for body in bodies:
                    chunk_candidates.append((body, unique_parent_ids, heading_prefix))
    else:
        heading_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

        for block in fallback_segments:
            stripped_block = block.strip()
            if not stripped_block:
                continue
            heading_match = heading_pattern.match(stripped_block)
            if heading_match:
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
                sentences = _split_sentences(piece)
                if not sentences:
                    continue
                bodies = _chunkify(
                    sentences,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    hard_limit=hard_limit,
                )
                if not bodies:
                    continue
                parent_ids = [root_id] + [info["id"] for info in parent_stack]
                unique_parent_ids = list(
                    dict.fromkeys(pid for pid in parent_ids if pid)
                )
                for body in bodies:
                    chunk_candidates.append((body, unique_parent_ids, ""))

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
            chunk_meta = {
                "tenant_id": meta["tenant_id"],
                "case_id": meta.get("case_id"),
                "source": text_path,
                "hash": content_hash,
                "external_id": external_id,
                "content_hash": content_hash,
                "parent_ids": parent_ids,
            }
            if document_id:
                chunk_meta["document_id"] = document_id
                chunk_meta["doc_id"] = document_id
            if meta.get("embedding_profile"):
                chunk_meta["embedding_profile"] = meta["embedding_profile"]
            if meta.get("vector_space_id"):
                chunk_meta["vector_space_id"] = meta["vector_space_id"]
            if meta.get("collection_id"):
                chunk_meta["collection_id"] = meta["collection_id"]
            if meta.get("workflow_id"):
                chunk_meta["workflow_id"] = meta["workflow_id"]
            chunks.append(
                {
                    "content": chunk_text,
                    "normalized": normalised,
                    "meta": chunk_meta,
                }
            )

    for parent_id, info in parent_nodes.items():
        content_parts = parent_contents.get(parent_id) or []
        content_text = "\n\n".join(part for part in content_parts if part).strip()
        if content_text:
            info = dict(info)
            info["content"] = content_text
            parent_nodes[parent_id] = info

    limited_parents = limit_parent_payload(parent_nodes)
    payload = {"chunks": chunks, "parents": limited_parents}
    out_path = _build_path(meta, "embeddings", "chunks.json")
    object_store.write_json(out_path, payload)
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
@observe_span(name="ingestion.embed")
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, str]:
    """Generate embedding vectors for chunks via LiteLLM."""

    raw_chunks = object_store.read_json(chunks_path)
    parents: Dict[str, object] = {}
    if isinstance(raw_chunks, dict):
        chunks = list(raw_chunks.get("chunks", []) or [])
        parent_payload = raw_chunks.get("parents") or {}
        if isinstance(parent_payload, dict):
            parents = parent_payload
    else:
        chunks = list(raw_chunks or [])
    # Attach task context early for tracing
    try:
        update_observation(
            tags=["ingestion", "embed"],
            user_id=str(meta.get("tenant_id")) if meta.get("tenant_id") else None,
            session_id=str(meta.get("case_id")) if meta.get("case_id") else None,
            metadata={
                "embedding_profile": meta.get("embedding_profile"),
                "vector_space_id": meta.get("vector_space_id"),
                "collection_id": meta.get("collection_id"),
            },
        )
    except Exception:
        pass

    client = get_embedding_client()
    batch_size = max(
        1, int(getattr(settings, "EMBEDDINGS_BATCH_SIZE", client.batch_size))
    )

    prepared: List[Dict[str, object]] = []
    for ch in chunks:
        normalised = ch.get("normalized") or normalise_text(ch.get("content", ""))
        prepared.append({**ch, "normalized": normalised or ""})

    total_chunks = len(prepared)
    embeddings: List[Dict[str, object]] = []
    expected_dim: Optional[int] = None
    batches = 0

    for start in range(0, total_chunks, batch_size):
        batch = prepared[start : start + batch_size]
        if not batch:
            continue
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
        batch_dim: Optional[int] = len(result.vectors[0]) if result.vectors else None
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

        for entry, vector in zip(batch, result.vectors):
            if expected_dim is not None and len(vector) != expected_dim:
                raise ValueError("Embedding dimension mismatch")
            embeddings.append(
                {
                    **entry,
                    "embedding": list(vector),
                    "vector_dim": len(vector),
                }
            )

    logger.info(
        "ingestion.embed.summary",
        extra={"chunks": total_chunks, "batches": batches},
    )
    payload = {"chunks": embeddings, "parents": parents}
    out_path = _build_path(meta, "embeddings", "vectors.json")
    object_store.write_json(out_path, payload)
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
def upsert(
    meta: Dict[str, str],
    embeddings_path: str,
    tenant_schema: Optional[str] = None,
) -> int:
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
                    "tenant_id": meta.get("tenant_id") if meta else None,
                    "case_id": meta.get("case_id") if meta else None,
                    "chunk_index": index,
                    "keys": (
                        sorted(raw_meta.keys()) if isinstance(raw_meta, dict) else None
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
                    "vector_space_id",
                    "process",
                    "collection_id",
                    "workflow_id",
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

    tenant_id: Optional[str] = meta.get("tenant_id") if meta else None
    if not tenant_id:
        tenant_id = next(
            (
                str(chunk.meta.get("tenant_id"))
                for chunk in chunk_objs
                if chunk.meta and chunk.meta.get("tenant_id")
            ),
            None,
        )
    if not tenant_id:
        raise ValueError("tenant_id required for upsert")

    for chunk in chunk_objs:
        chunk_tenant = chunk.meta.get("tenant_id") if chunk.meta else None
        if chunk_tenant and str(chunk_tenant) != tenant_id:
            raise ValueError("chunk tenant mismatch")

    expected_dimension_value = meta.get("vector_space_dimension") if meta else None
    expected_dimension: Optional[int] = None
    if expected_dimension_value is not None:
        try:
            expected_dimension = int(expected_dimension_value)
        except (TypeError, ValueError):
            expected_dimension = None

    ensure_embedding_dimensions(
        chunk_objs,
        expected_dimension,
        tenant_id=tenant_id,
        process=meta.get("process") if meta else None,
        workflow_id=meta.get("workflow_id") if meta else None,
        embedding_profile=meta.get("embedding_profile") if meta else None,
        vector_space_id=meta.get("vector_space_id") if meta else None,
    )

    schema = tenant_schema or (meta.get("tenant_schema") if meta else None)

    router = get_default_router()
    tenant_client = router
    for_tenant = getattr(router, "for_tenant", None)
    if callable(for_tenant):
        try:
            tenant_client = for_tenant(tenant_id, schema)
        except TypeError:
            tenant_client = for_tenant(tenant_id)
    written = tenant_client.upsert_chunks(chunk_objs)
    return written


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
def ingestion_run(
    tenant_id: str,
    case_id: str,
    document_ids: List[str],
    priority: str = "normal",
    trace_id: str | None = None,
) -> Dict[str, object]:
    """Placeholder ingestion dispatcher used by the ingestion run endpoint."""

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
    return {"status": "queued", "queued_at": queued_at}
