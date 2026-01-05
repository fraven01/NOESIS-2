from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import time
import uuid
import warnings
from datetime import datetime
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

from celery import shared_task

from ai_core.graphs.transition_contracts import (
    GraphTransition,
    StandardTransitionResult,
)
from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.ids.http_scope import normalize_task_context
from documents.api import normalize_from_raw
from documents.contracts import NormalizedDocument, NormalizedDocumentInputV1
from documents.pipeline import (
    DocumentPipelineConfig,
    DocumentProcessingContext,
    DocumentProcessingMetadata,
    ParsedResult,
    ParsedTextBlock,
)
from ai_core.infra import observability as observability_helpers
from ai_core.infra.observability import (
    emit_event,
    observe_span,
    record_span,
    tracing_enabled,
    update_observation,
)
from ai_core.infra.config import get_config
from ai_core.ingestion_orchestration import (
    IngestionContextBuilder,
    ObservabilityWrapper,
)
from ai_core.tools.errors import RateLimitedError
from common.celery import RetryableTask, ScopedTask
from common.logging import get_logger
from django.conf import settings
from django.utils import timezone
from pydantic import ValidationError
from redis import Redis

from .infra import object_store, pii
from .infra.serialization import to_jsonable
from .infra.pii_flags import get_pii_config
from .segmentation import segment_markdown_blocks
from .rag import metrics
from .rag.embedding_config import (
    build_embedding_model_version,
    build_vector_space_id,
    get_embedding_profile,
)
from .rag.embedding_cache import (
    compute_text_hash,
    fetch_cached_embeddings,
    store_cached_embeddings,
)
from .rag.semantic_chunker import SectionChunkPlan, SemanticChunker, SemanticTextBlock
from .rag.parents import limit_parent_payload
from .rag.schemas import Chunk
from .rag.normalization import normalise_text
from .rag.ingestion_contracts import ChunkMeta, ensure_embedding_dimensions
from .rag.chunking import RoutingAwareChunker
from .rag.embeddings import (
    EmbeddingBatchResult,
    EmbeddingClientError,
    get_embedding_client,
)
from .rag.pricing import calculate_embedding_cost
from .rag.vector_store import get_default_router
from .rag.vector_space_resolver import resolve_vector_space_full
from .rag.vector_client import get_client_for_schema, get_default_schema

_ZERO_EPSILON = 1e-12
_FAILED_CHUNK_ID_LIMIT = 10


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

_DEDUPE_TTL_SECONDS = 24 * 60 * 60
_CACHE_TTL_CHUNK_SECONDS = 60 * 60
_CACHE_TTL_EMBED_SECONDS = 24 * 60 * 60


def _coerce_cache_part(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
    else:
        try:
            candidate = str(value).strip()
        except Exception:
            return None
    return candidate or None


def _hash_parts(*parts: Any) -> Optional[str]:
    text_parts: List[str] = []
    for part in parts:
        candidate = _coerce_cache_part(part)
        if not candidate:
            return None
        text_parts.append(candidate)
    payload = "|".join(text_parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_embedding_profile_id(
    meta: Mapping[str, Any], *, allow_default: bool
) -> Optional[str]:
    profile_id = _coerce_cache_part(meta.get("embedding_profile"))
    if profile_id:
        return profile_id
    if not allow_default:
        return None
    return _coerce_cache_part(getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", None))


def _resolve_embedding_model_version(meta: Mapping[str, Any]) -> Optional[str]:
    profile_id = _resolve_embedding_profile_id(meta, allow_default=True)
    if not profile_id:
        return None
    try:
        profile = get_embedding_profile(profile_id)
    except Exception:
        return None
    return build_embedding_model_version(profile)


def _resolve_vector_space_id(meta: Mapping[str, Any]) -> Optional[str]:
    profile_id = _resolve_embedding_profile_id(meta, allow_default=True)
    if not profile_id:
        return None
    try:
        profile = get_embedding_profile(profile_id)
    except Exception:
        return None
    return build_vector_space_id(profile.id, profile.model_version)


def _resolve_vector_space_schema(meta: Mapping[str, Any]) -> Optional[str]:
    profile_id = _resolve_embedding_profile_id(meta, allow_default=True)
    if not profile_id:
        return None
    try:
        resolution = resolve_vector_space_full(profile_id)
    except Exception:
        return None
    return resolution.vector_space.schema


def _resolve_redis_url() -> Optional[str]:
    try:
        url = get_config().redis_url
    except Exception:
        url = getattr(settings, "REDIS_URL", None) or getattr(
            settings, "CELERY_BROKER_URL", None
        )
    return _coerce_cache_part(url)


def _redis_client() -> Optional[Redis]:
    url = _resolve_redis_url()
    if not url:
        return None
    try:
        client = Redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception as exc:
        logger.warning("task.redis.unavailable", extra={"error": str(exc)})
        return None


def _cache_key(task_name: str, idempotency_key: str) -> str:
    return f"task:cache:{task_name}:{idempotency_key}"


def _dedupe_key(task_name: str, idempotency_key: str) -> str:
    return f"task:dedupe:{task_name}:{idempotency_key}"


def _cache_get(client: Redis, key: str) -> Optional[str]:
    try:
        value = client.get(key)
    except Exception:
        return None
    return _coerce_cache_part(value)


def _cache_set(client: Redis, key: str, value: str, ttl_seconds: int) -> None:
    try:
        client.set(key, value, ex=int(ttl_seconds))
    except Exception:
        return None


def _cache_delete(client: Redis, key: str) -> None:
    try:
        client.delete(key)
    except Exception:
        return None


def _dedupe_status(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value.startswith("done:"):
        return "done"
    if value.startswith("inflight:"):
        return "inflight"
    return "inflight"


def _acquire_dedupe_lock(client: Redis, key: str, ttl_seconds: int, token: str) -> bool:
    try:
        return bool(client.set(key, f"inflight:{token}", nx=True, ex=int(ttl_seconds)))
    except Exception:
        return False


def _mark_dedupe_done(client: Redis, key: str, ttl_seconds: int, token: str) -> None:
    try:
        client.set(key, f"done:{token}", ex=int(ttl_seconds))
    except Exception:
        return None


def _release_dedupe_lock(client: Redis, key: str, token: str) -> None:
    try:
        current = client.get(key)
    except Exception:
        return None
    if current == f"inflight:{token}":
        try:
            client.delete(key)
        except Exception:
            return None


def _object_store_path_exists(path: str) -> bool:
    try:
        return (object_store.BASE_PATH / path).exists()
    except Exception:
        return False


def _task_context_payload(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, MappingABC):
        return {}
    try:
        context = tool_context_from_meta(meta)
    except Exception:
        return {}
    payload: Dict[str, Any] = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
    }
    if context.scope.run_id:
        payload["run_id"] = context.scope.run_id
    if context.scope.ingestion_run_id:
        payload["ingestion_run_id"] = context.scope.ingestion_run_id
    return payload


def _log_cache_hit(
    *,
    task_name: str,
    idempotency_key: str,
    cache_key: str,
    cached_path: str,
    meta: Optional[Mapping[str, Any]],
) -> None:
    payload = {
        "task_name": task_name,
        "idempotency_key": idempotency_key,
        "cache_key": cache_key,
        "path": cached_path,
        "cache_hit": True,
        **_task_context_payload(meta),
    }
    logger.info("task.cache.hit", extra=payload)
    emit_event("task.cache.hit", payload)


def _log_embedding_cache_hit(
    *,
    task_name: str,
    model_version: str,
    hit_count: int,
    total_chunks: int,
    meta: Optional[Mapping[str, Any]],
) -> None:
    payload = {
        "task_name": task_name,
        "model_version": model_version,
        "cache_hit_count": hit_count,
        "chunks_total": total_chunks,
        "cache_hit": True,
        **_task_context_payload(meta),
    }
    logger.info("rag.embedding_cache.hit", extra=payload)
    emit_event("rag.embedding_cache.hit", payload)


def _log_dedupe_hit(
    *,
    task_name: str,
    idempotency_key: str,
    dedupe_key: str,
    status: str,
    meta: Optional[Mapping[str, Any]],
) -> None:
    payload = {
        "task_name": task_name,
        "idempotency_key": idempotency_key,
        "dedupe_key": dedupe_key,
        "dedupe_status": status,
        **_task_context_payload(meta),
    }
    logger.info("task.dedupe.hit", extra=payload)
    emit_event("task.dedupe.hit", payload)


def _extract_chunk_meta_value(chunks: Iterable[Any], key: str) -> Optional[str]:
    for entry in chunks:
        if not isinstance(entry, MappingABC):
            continue
        meta = entry.get("meta")
        if not isinstance(meta, MappingABC):
            continue
        value = _coerce_cache_part(meta.get(key))
        if value:
            return value
    return None


def _resolve_upsert_content_hash(
    meta: Optional[Mapping[str, Any]],
    chunks: Iterable[Any],
) -> Optional[str]:
    if isinstance(meta, MappingABC):
        value = _coerce_cache_part(meta.get("content_hash"))
        if value:
            return value
    return _extract_chunk_meta_value(chunks, "content_hash")


def _resolve_upsert_vector_space_id(
    meta: Optional[Mapping[str, Any]],
    chunks: Iterable[Any],
) -> Optional[str]:
    if isinstance(meta, MappingABC):
        value = _coerce_cache_part(meta.get("vector_space_id"))
        if value:
            return value
    return _extract_chunk_meta_value(chunks, "vector_space_id")


def _resolve_upsert_embedding_profile(
    meta: Optional[Mapping[str, Any]],
    chunks: Iterable[Any],
) -> Optional[str]:
    if isinstance(meta, MappingABC):
        value = _coerce_cache_part(meta.get("embedding_profile"))
        if value:
            return value
    value = _extract_chunk_meta_value(chunks, "embedding_profile")
    if value:
        return value
    return _coerce_cache_part(getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", None))


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    """Build object store path with tenant and case identifiers.

    BREAKING CHANGE (Option A - Strict Separation):
    case_id is a business identifier, extracted from business_context.
    """
    context = tool_context_from_meta(meta)
    tenant = object_store.sanitize_identifier(context.scope.tenant_id)
    # BREAKING CHANGE: Extract case_id from business_context, not scope_context
    case = object_store.sanitize_identifier(context.business.case_id or "upload")
    return "/".join([tenant, case, *parts])


def _resolve_artifact_filename(meta: Mapping[str, Any], kind: str) -> str:
    """Derive a per-document filename for chunking artifacts."""

    def _normalise(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        candidate = str(value).strip()
        if not candidate:
            return None
        try:
            return object_store.sanitize_identifier(candidate)
        except Exception:
            return None

    hash_seed: Optional[str] = None
    for key in ("content_hash", "hash"):
        hash_seed = _normalise(meta.get(key))
        if hash_seed:
            break

    identifier_seed: Optional[str] = None
    for key in ("external_id", "document_id"):
        identifier_seed = _normalise(meta.get(key))
        if identifier_seed:
            break

    seeds = [component for component in (hash_seed, identifier_seed) if component]
    if not seeds:
        seeds = [_normalise(meta.get("id"))]
    seeds = [component for component in seeds if component]

    if not seeds:
        seeds = [uuid.uuid4().hex]

    seed = "-".join(seeds)
    base_name = f"{kind}-{seed}.json"
    try:
        return object_store.safe_filename(base_name)
    except Exception:
        return object_store.safe_filename(f"{kind}-{uuid.uuid4().hex}.json")


class _EmbedSpanMetrics:
    """Accumulate Langfuse span metadata for embedding phases."""

    def __init__(self) -> None:
        self.metadata: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        if value is None:
            return
        self.metadata[key] = value

    def add(self, key: str, value: float | int) -> None:
        if value is None:
            return
        current = self.metadata.get(key, 0.0)
        try:
            current_value = float(current)
        except (TypeError, ValueError):
            current_value = 0.0
        self.metadata[key] = current_value + float(value)

    def finalise(self) -> None:
        if self.metadata:
            update_observation(metadata=self.metadata)


def _observed_embed_section(name: str) -> Iterator[_EmbedSpanMetrics]:
    metrics = _EmbedSpanMetrics()

    span_cm = nullcontext()
    if tracing_enabled():
        get_tracer = getattr(observability_helpers, "_get_tracer", None)
        tracer = None
        if callable(get_tracer):
            try:
                tracer = get_tracer()
            except Exception:
                tracer = None
        if tracer is not None:
            try:
                candidate = tracer.start_as_current_span(f"ingestion.embed.{name}")
            except Exception:
                candidate = None
            else:
                if candidate is not None:
                    span_cm = candidate

    @contextmanager
    def _span() -> Iterator[_EmbedSpanMetrics]:
        try:
            with span_cm:
                try:
                    yield metrics
                except Exception:
                    metrics.set("status", "error")
                    raise
        finally:
            metrics.finalise()

    return _span()


def _extract_chunk_identifier(entry: Dict[str, Any]) -> Optional[str]:
    meta = entry.get("meta")
    if isinstance(meta, dict):
        for key in ("chunk_id", "hash", "document_id", "external_id"):
            value = meta.get(key)
            if value:
                return str(value)
    value = entry.get("id")
    if value:
        return str(value)
    return None


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
    case_status: Optional[str] = None,
    case_phase: Optional[str] = None,
    collection_scope: Optional[str] = None,
    document_collection_key: Optional[str] = None,
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
    if case_status:
        extra["case_status"] = case_status
    if case_phase:
        extra["case_phase"] = case_phase
    if collection_scope:
        extra["collection_scope"] = collection_scope
    if document_collection_key:
        extra["document_collection_key"] = document_collection_key
    logger.info("ingestion.start", extra=extra)
    if trace_id:
        record_span(
            "rag.ingestion.run.start",
            trace_id=trace_id,
            attributes={**extra},
        )


def _jsonify_for_task(value: Any) -> Any:
    """Convert objects returned by ingestion tasks into JSON primitives."""
    return to_jsonable(value)


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
    case_status: Optional[str] = None,
    case_phase: Optional[str] = None,
    collection_scope: Optional[str] = None,
    document_collection_key: Optional[str] = None,
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
    if case_status:
        extra["case_status"] = case_status
    if case_phase:
        extra["case_phase"] = case_phase
    if collection_scope:
        extra["collection_scope"] = collection_scope
    if document_collection_key:
        extra["document_collection_key"] = document_collection_key
    logger.info("ingestion.end", extra=extra)
    metrics.INGESTION_RUN_MS.observe(float(duration_ms))
    if trace_id:
        record_span(
            "rag.ingestion.run.end",
            trace_id=trace_id,
            attributes={**extra},
        )


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
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


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
def extract_text(meta: Dict[str, str], raw_path: str) -> Dict[str, str]:
    """Decode bytes to text and store."""
    full = object_store.BASE_PATH / raw_path
    text = full.read_bytes().decode("utf-8")
    out_path = _build_path(meta, "text", f"{Path(raw_path).stem}.txt")
    object_store.put_bytes(out_path, text.encode("utf-8"))
    return {"path": out_path}


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
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


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
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
    context = tool_context_from_meta(meta)
    doc_type = str(
        meta.get("doc_class")
        or context.business.collection_id
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


_PARSED_BLOCK_KINDS = {
    "paragraph",
    "heading",
    "list",
    "table_summary",
    "slide",
    "note",
    "code",
    "other",
}


def _coerce_block_kind(value: object) -> str:
    if value is None:
        return "paragraph"
    candidate = str(value).strip().lower()
    return candidate if candidate in _PARSED_BLOCK_KINDS else "paragraph"


def _coerce_section_path(value: object) -> Optional[Tuple[str, ...]]:
    if isinstance(value, (list, tuple)):
        path = tuple(str(part).strip() for part in value if str(part).strip())
        return path or None
    return None


def _build_parsed_blocks(
    *,
    text: str,
    structured_blocks: Sequence[Mapping[str, object]],
    mask_fn: Callable[[str], str],
) -> List[ParsedTextBlock]:
    blocks: List[ParsedTextBlock] = []
    if structured_blocks:
        for block in structured_blocks:
            text_value = block.get("text") if isinstance(block, Mapping) else None
            if text_value is None:
                continue
            text_str = str(text_value).strip()
            if not text_str:
                continue
            kind = _coerce_block_kind(block.get("kind"))
            section_path = _coerce_section_path(block.get("section_path"))
            page_index = None
            raw_page = block.get("page_index")
            if raw_page is not None:
                try:
                    page_index = int(raw_page)
                except (TypeError, ValueError):
                    page_index = None
            table_meta = None
            raw_table = block.get("table_meta")
            if isinstance(raw_table, Mapping):
                table_meta = raw_table
            language = None
            raw_language = block.get("language")
            if isinstance(raw_language, str) and raw_language.strip():
                language = raw_language.strip()
            try:
                blocks.append(
                    ParsedTextBlock(
                        text=mask_fn(text_str),
                        kind=kind,
                        section_path=section_path,
                        page_index=page_index,
                        table_meta=table_meta,
                        language=language,
                    )
                )
            except ValueError:
                continue
        return blocks

    heading_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
    current_path: List[str] = []
    for segment in segment_markdown_blocks(text):
        stripped = segment.strip()
        if not stripped:
            continue
        heading_match = heading_pattern.match(stripped)
        if heading_match:
            hashes, heading_title = heading_match.groups()
            level = len(hashes)
            title = heading_title.strip()
            if not title:
                continue
            while len(current_path) >= level:
                current_path.pop()
            current_path.append(title)
            try:
                blocks.append(
                    ParsedTextBlock(
                        text=mask_fn(title),
                        kind="heading",
                        section_path=tuple(current_path),
                    )
                )
            except ValueError:
                continue
            continue
        try:
            blocks.append(
                ParsedTextBlock(
                    text=mask_fn(stripped),
                    kind="paragraph",
                    section_path=tuple(current_path) if current_path else None,
                )
            )
        except ValueError:
            continue

    if not blocks and text.strip():
        try:
            blocks.append(ParsedTextBlock(text=mask_fn(text.strip()), kind="paragraph"))
        except ValueError:
            pass
    return blocks


def _build_processing_context(
    *,
    meta: Mapping[str, object],
    tool_context: Any,
) -> Optional[DocumentProcessingContext]:
    workflow_id = meta.get("workflow_id") or getattr(
        tool_context.business, "workflow_id", None
    )
    if workflow_id is None or str(workflow_id).strip() == "":
        return None
    document_id = meta.get("document_id") or getattr(
        tool_context.business, "document_id", None
    )
    if document_id is None or str(document_id).strip() == "":
        return None
    try:
        document_uuid = uuid.UUID(str(document_id))
    except (TypeError, ValueError, AttributeError):
        return None

    metadata = DocumentProcessingMetadata(
        tenant_id=str(tool_context.scope.tenant_id),
        collection_id=getattr(tool_context.business, "collection_id", None),
        case_id=getattr(tool_context.business, "case_id", None),
        workflow_id=str(workflow_id),
        document_id=document_uuid,
        source=str(meta.get("source")) if meta.get("source") else None,
        created_at=timezone.now(),
        trace_id=getattr(tool_context.scope, "trace_id", None),
    )
    return DocumentProcessingContext(
        metadata=metadata,
        trace_id=metadata.trace_id,
        span_id=metadata.span_id,
    )


@shared_task(
    base=RetryableTask,
    queue="ingestion",
    accepts_scope=True,
    time_limit=600,
    soft_time_limit=540,
)
@observe_span(name="ingestion.chunk")
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
                    return {"path": cached_path}
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
        return {"path": out_path}

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

    semantic_plans: List[SectionChunkPlan] = []
    if structured_blocks:
        warnings.warn(
            "SemanticChunker is deprecated and will be removed in a future version. "
            "Use HybridChunker from ai_core.rag.chunking instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        semantic_chunker = SemanticChunker(
            sentence_splitter=_split_sentences,
            chunkify_fn=lambda sentences: _chunkify(
                sentences,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                hard_limit=hard_limit,
            ),
        )
        semantic_blocks: List[SemanticTextBlock] = []
        for block in structured_blocks:
            text_value = str(block.get("text") or "")
            kind = str(block.get("kind") or "").lower() or "paragraph"
            section_path_raw = block.get("section_path")
            path_tuple: Tuple[str, ...] = ()
            if isinstance(section_path_raw, (list, tuple)):
                path_tuple = tuple(
                    str(part).strip()
                    for part in section_path_raw
                    if isinstance(part, str) and str(part).strip()
                )
            masked_text = _mask_for_chunk(text_value)
            semantic_blocks.append(
                SemanticTextBlock(
                    text=masked_text,
                    kind=kind,
                    section_path=path_tuple,
                )
            )
        semantic_plans = semantic_chunker.build_plans(semantic_blocks)

    if semantic_plans:
        section_registry: Dict[Tuple[str, ...], Dict[str, object]] = {}
        for plan in semantic_plans:
            parent_infos: List[Dict[str, object]] = []
            if plan.path:
                for depth in range(1, len(plan.path) + 1):
                    prefix_tuple = plan.path[:depth]
                    info = section_registry.get(prefix_tuple)
                    if info is None:
                        title_candidate = prefix_tuple[-1] if prefix_tuple else ""
                        info = _register_section(title_candidate, depth, prefix_tuple)
                        section_registry[prefix_tuple] = info
                    parent_infos.append(info)

            target_parent_id = parent_infos[-1]["id"] if parent_infos else root_id
            if plan.parent_text:
                _append_parent_text_with_root(
                    target_parent_id, plan.parent_text, plan.level
                )

            parent_ids = [root_id] + [info["id"] for info in parent_infos]
            unique_parent_ids = list(dict.fromkeys(pid for pid in parent_ids if pid))
            for body in plan.chunk_bodies:
                body_text = body.strip()
                if not body_text:
                    continue
                chunk_candidates.append(
                    (body_text, unique_parent_ids, plan.heading_prefix)
                )
            last_registered_stack = parent_infos
    else:
        pending_pieces = []
        pending_parent_ids: Optional[Tuple[str, ...]] = None

        def _flush_pending() -> None:
            nonlocal pending_pieces, pending_parent_ids
            if not pending_pieces or pending_parent_ids is None:
                pending_pieces = []
                pending_parent_ids = None
                return

            combined_text = "\n\n".join(part for part in pending_pieces if part.strip())
            sentences = _split_sentences(combined_text)
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
                unique_parent_ids = list(
                    dict.fromkeys(pid for pid in parent_ids if pid)
                )
                new_parent_ids = tuple(unique_parent_ids)

                if (
                    pending_parent_ids is not None
                    and pending_parent_ids != new_parent_ids
                ):
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
    return {"path": out_path}


@shared_task(
    base=RetryableTask,
    queue="ingestion",
    accepts_scope=True,
    time_limit=300,
    soft_time_limit=270,
)
@observe_span(name="ingestion.embed")
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, str]:
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
                    return {"path": cached_path}
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
                        return {"path": cached_path}
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
                user_id=(
                    str(context.scope.tenant_id) if context.scope.tenant_id else None
                ),
                session_id=(
                    str(context.business.case_id) if context.business.case_id else None
                ),
                metadata={
                    "embedding_profile": meta.get("embedding_profile"),
                    "vector_space_id": resolved_vector_space_id,
                    "embedding_model_version": embedding_model_version,
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
        return {"path": out_path}
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


@shared_task(base=RetryableTask, queue="ingestion", accepts_scope=True)
def upsert(
    meta: Dict[str, str],
    embeddings_path: str,
    tenant_schema: Optional[str] = None,
    *,
    vector_client: Optional[Any] = None,
    vector_client_factory: Optional[Callable[[], Any]] = None,
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

    cache_client = _redis_client()
    dedupe_key = None
    dedupe_token = None
    idempotency_key = None
    if cache_client is not None:
        context = tool_context_from_meta(meta) if meta else None
        tenant_id = context.scope.tenant_id if context else None
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
                return 0
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
                    return 0
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
        context = tool_context_from_meta(meta) if meta else None
        try:
            logger.warning(
                "ingestion.upsert.parents_loaded",
                extra={
                    "event": "DEBUG.TASKS.UPSERT.PARENTS_LOADED",
                    "tenant_id": context.scope.tenant_id if context else None,
                    "case_id": context.business.case_id if context else None,
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
                context = tool_context_from_meta(meta) if meta else None
                logger.error(
                    "ingestion.chunk.meta.invalid",
                    extra={
                        "tenant_id": context.scope.tenant_id if context else None,
                        "case_id": context.business.case_id if context else None,
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

        context = tool_context_from_meta(meta) if meta else None
        tenant_id: Optional[str] = context.scope.tenant_id if context else None
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

        # BREAKING CHANGE (Option A): workflow_id from business_context
        business_context = context.business if context else None
        ensure_embedding_dimensions(
            chunk_objs,
            expected_dimension,
            tenant_id=tenant_id,
            process=meta.get("process") if meta else None,
            workflow_id=business_context.workflow_id if business_context else None,
            embedding_profile=meta.get("embedding_profile") if meta else None,
            vector_space_id=meta.get("vector_space_id") if meta else None,
        )

        schema = tenant_schema or (meta.get("tenant_schema") if meta else None)
        vector_space_schema = _resolve_vector_space_schema(meta) if meta else None

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
        return written
    except Exception:
        if cache_client is not None and dedupe_key and dedupe_token:
            _release_dedupe_lock(cache_client, dedupe_key, dedupe_token)
        raise


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


def _is_redis_broker(url: str) -> bool:
    return url.startswith("redis://") or url.startswith("rediss://")


def _resolve_dlq_queue_key(queue_name: str) -> str:
    prefix = getattr(settings, "CELERY_REDIS_QUEUE_PREFIX", "") or ""
    return f"{prefix}{queue_name}"


def _decode_dlq_message(raw: bytes) -> Dict[str, Any] | None:
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_dead_lettered_at(payload: Mapping[str, Any]) -> Optional[float]:
    args: Any = None
    body = payload.get("body")
    if isinstance(body, (list, tuple)) and body:
        args = body[0]
    elif isinstance(body, Mapping):
        args = body.get("args")
    if args is None:
        args = payload.get("args")

    candidate: Any = None
    if isinstance(args, (list, tuple)) and args:
        candidate = args[0]
    elif isinstance(args, Mapping):
        candidate = args

    if not isinstance(candidate, Mapping):
        return None
    timestamp = candidate.get("dead_lettered_at")
    if timestamp is None:
        return None
    try:
        return float(timestamp)
    except (TypeError, ValueError):
        return None


def _is_dlq_message_expired(raw: bytes, cutoff_ts: float) -> bool:
    payload = _decode_dlq_message(raw)
    if payload is None:
        return False
    dead_lettered_at = _extract_dead_lettered_at(payload)
    if dead_lettered_at is None:
        return False
    return dead_lettered_at < cutoff_ts


@shared_task(
    base=ScopedTask,
    queue="default",
    name="ai_core.tasks.cleanup_dead_letter",
)
def cleanup_dead_letter_queue(
    *,
    max_messages: int = 1000,
    ttl_ms: Optional[int] = None,
    queue_name: str = "dead_letter",
) -> Dict[str, int | str]:
    """Purge expired dead-letter tasks from a Redis-backed queue."""

    broker_url = str(getattr(settings, "CELERY_BROKER_URL", "") or "")
    if not _is_redis_broker(broker_url):
        return {"status": "skipped", "reason": "non_redis_broker"}

    effective_ttl = ttl_ms
    if effective_ttl is None:
        effective_ttl = getattr(settings, "CELERY_DLQ_TTL_MS", 0)
    try:
        ttl_value = int(effective_ttl) if effective_ttl is not None else 0
    except (TypeError, ValueError):
        ttl_value = 0
    if ttl_value <= 0:
        return {"status": "skipped", "reason": "ttl_disabled"}

    cutoff_ts = time.time() - (ttl_value / 1000.0)
    queue_key = _resolve_dlq_queue_key(queue_name)

    removed = 0
    kept = 0
    scanned = 0

    client = Redis.from_url(broker_url)
    scan_limit = max(0, int(max_messages))
    try:
        queue_length = int(client.llen(queue_key))
    except Exception:
        queue_length = 0
    if queue_length:
        scan_limit = min(scan_limit, queue_length)
    for _ in range(scan_limit):
        raw = client.lpop(queue_key)
        if raw is None:
            break
        scanned += 1
        if _is_dlq_message_expired(raw, cutoff_ts):
            removed += 1
            continue
        client.rpush(queue_key, raw)
        kept += 1

    logger.info(
        "dlq.cleanup.completed",
        extra={
            "queue": queue_name,
            "scanned": scanned,
            "removed": removed,
            "kept": kept,
            "ttl_ms": ttl_value,
        },
    )
    return {
        "status": "ok",
        "scanned": scanned,
        "removed": removed,
        "kept": kept,
    }


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


@shared_task(
    base=ScopedTask,
    queue="default",
    name="ai_core.tasks.alert_dead_letter",
)
def alert_dead_letter_queue(
    *,
    threshold: Optional[int] = None,
    queue_name: str = "dead_letter",
) -> Dict[str, int | str | bool]:
    """Emit a structured alert when the Redis DLQ exceeds the threshold."""

    broker_url = str(getattr(settings, "CELERY_BROKER_URL", "") or "")
    if not _is_redis_broker(broker_url):
        return {"status": "skipped", "reason": "non_redis_broker"}

    threshold_value = threshold
    if threshold_value is None:
        threshold_value = getattr(settings, "CELERY_DLQ_ALERT_THRESHOLD", 10)
    threshold_value = _coerce_positive_int(threshold_value, 10)
    if threshold_value <= 0:
        return {"status": "skipped", "reason": "threshold_disabled"}

    queue_key = _resolve_dlq_queue_key(queue_name)
    client = Redis.from_url(broker_url)
    try:
        queue_length = int(client.llen(queue_key))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "dlq.alert.redis_error",
            extra={"queue": queue_name, "error": str(exc)},
        )
        return {"status": "error", "reason": "redis_error"}

    payload = {
        "queue": queue_name,
        "queue_length": queue_length,
        "threshold": threshold_value,
    }
    if queue_length > threshold_value:
        emit_event("dlq.threshold_exceeded", payload)
        logger.warning("dlq.threshold_exceeded", extra=payload)

    return {
        "status": "ok",
        "queue_length": queue_length,
        "threshold": threshold_value,
        "alerted": queue_length > threshold_value,
    }


def _resolve_event_emitter(meta: Optional[Mapping[str, Any]] = None):
    if isinstance(meta, MappingABC):
        candidate = meta.get("ingestion_event_emitter")
        if callable(candidate):
            return candidate
    return None


def _callable_accepts_kwarg(func: Any, keyword: str) -> bool:
    """Return True when the callable supports the provided keyword argument."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - non introspectable callables
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _build_ingestion_graph(event_emitter: Optional[Any]):
    """Invoke build_universal_ingestion_graph."""
    from ai_core.graphs.technical.universal_ingestion_graph import (
        build_universal_ingestion_graph,
    )

    return build_universal_ingestion_graph()


def build_graph(*, event_emitter: Optional[Any] = None):
    """Legacy shim so older tests can import ai_core.tasks.build_graph."""

    return _build_ingestion_graph(event_emitter)


def _coerce_str(value: Any | None) -> Optional[str]:
    """Return a stripped string representation when possible."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    try:
        text = str(value)
    except Exception:
        return None
    return text.strip() or None


def _extract_from_mapping(mapping: Any, key: str) -> Optional[str]:
    if not isinstance(mapping, MappingABC):
        return None
    return _coerce_str(mapping.get(key))


def _resolve_document_id(
    state: Mapping[str, Any], meta: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Try to resolve the document identifier from meta/state payloads.

    BREAKING CHANGE (Option A - Strict Separation):
    Document ID is a business identifier, so we check business_context first.
    """

    state_meta = state.get("meta") if isinstance(state, MappingABC) else None
    tool_context = None
    if isinstance(meta, MappingABC):
        try:
            tool_context = tool_context_from_meta(meta)
        except (TypeError, ValueError):
            tool_context = None

    # Check business_context first (BREAKING CHANGE)
    candidate = tool_context.business.document_id if tool_context is not None else None
    if candidate:
        return _coerce_str(candidate)

    # Fallback to other sources
    for source in (meta, state_meta, state):
        candidate = _extract_from_mapping(source, "document_id")
        if candidate:
            return candidate

    raw_document = state.get("raw_document") if isinstance(state, MappingABC) else None
    if isinstance(raw_document, MappingABC):
        for key in ("document_id", "external_id", "id"):
            candidate = _coerce_str(raw_document.get(key))
            if candidate:
                return candidate
        raw_metadata = raw_document.get("metadata")
        if isinstance(raw_metadata, MappingABC):
            for key in ("document_id", "external_id", "id"):
                candidate = _coerce_str(raw_metadata.get(key))
                if candidate:
                    return candidate

    return None


def _resolve_trace_context(
    state: Mapping[str, Any], meta: Optional[Mapping[str, Any]]
) -> Dict[str, Optional[str]]:
    """Collect identifiers required for tracing metadata.

    BREAKING CHANGE (Option A - Strict Separation):
    Business IDs (case_id, workflow_id) now extracted from business_context,
    not scope_context.
    """

    state_meta = state.get("meta") if isinstance(state, MappingABC) else None
    tool_context = None
    if isinstance(meta, MappingABC):
        try:
            tool_context = tool_context_from_meta(meta)
        except (TypeError, ValueError):
            tool_context = None
    scope_context = tool_context.scope if tool_context else None
    business_context = tool_context.business if tool_context else None

    # Infrastructure IDs from scope_context
    tenant_id = _coerce_str(
        (scope_context.tenant_id if scope_context else None)
        or _extract_from_mapping(state_meta, "tenant_id")
        or _extract_from_mapping(state, "tenant_id")
    )
    trace_id = _coerce_str(
        (scope_context.trace_id if scope_context else None)
        or _extract_from_mapping(state_meta, "trace_id")
        or _extract_from_mapping(state, "trace_id")
    )

    # Business IDs from business_context (BREAKING CHANGE)
    case_id = _coerce_str(
        (business_context.case_id if business_context else None)
        or _extract_from_mapping(state_meta, "case_id")
        or _extract_from_mapping(state, "case_id")
    )
    workflow_id = _coerce_str(
        (business_context.workflow_id if business_context else None)
        or _extract_from_mapping(state_meta, "workflow_id")
        or _extract_from_mapping(state, "workflow_id")
    )
    document_id = _resolve_document_id(state, meta)

    return {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "workflow_id": workflow_id,
        "trace_id": trace_id,
        "document_id": document_id,
    }


def _build_base_span_metadata(
    trace_context: Mapping[str, Optional[str]],
    graph_run_id: Optional[str],
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    for key in (
        "tenant_id",
        "case_id",
        "trace_id",
        "document_id",
        "workflow_id",
        "run_id",
        "ingestion_run_id",
        "collection_id",
        "document_version_id",
    ):
        value = _coerce_str(trace_context.get(key))
        if value:
            attributes[f"meta.{key}"] = value
    if graph_run_id:
        attributes["meta.graph_run_id"] = graph_run_id
    return attributes


def _coerce_transition_result(
    transition: object,
) -> StandardTransitionResult | None:
    if isinstance(transition, GraphTransition):
        return transition.result
    if isinstance(transition, StandardTransitionResult):
        return transition
    if isinstance(transition, MappingABC):
        try:
            return StandardTransitionResult.model_validate(transition)
        except ValidationError:
            return None
    return None


def _collect_transition_attributes(
    transition: StandardTransitionResult, phase: str
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    pipeline_phase = None
    if transition.pipeline and transition.pipeline.phase:
        pipeline_phase = _coerce_str(transition.pipeline.phase)
    phase_value = pipeline_phase or _coerce_str(transition.phase)
    attributes["meta.phase"] = phase_value or phase

    severity_value = _coerce_str(transition.severity)
    if severity_value:
        attributes["meta.severity"] = severity_value

    context_payload = transition.context
    if isinstance(context_payload, MappingABC):
        document_id = _coerce_str(context_payload.get("document_id"))
        if document_id:
            attributes["meta.document_id"] = document_id
        run_id = _coerce_str(context_payload.get("run_id"))
        if run_id:
            attributes["meta.run_id"] = run_id
        ingestion_run_id = _coerce_str(context_payload.get("ingestion_run_id"))
        if ingestion_run_id:
            attributes["meta.ingestion_run_id"] = ingestion_run_id
        collection_id = _coerce_str(context_payload.get("collection_id"))
        if collection_id:
            attributes["meta.collection_id"] = collection_id
        document_version_id = _coerce_str(context_payload.get("document_version_id"))
        if document_version_id:
            attributes["meta.document_version_id"] = document_version_id

    if phase == "document_pipeline":
        pipeline_payload = transition.pipeline
        if pipeline_payload is not None:
            run_until = getattr(pipeline_payload, "run_until", None)
            run_until_label = None
            if run_until is not None:
                if hasattr(run_until, "value"):
                    run_until_label = _coerce_str(getattr(run_until, "value", None))
                if not run_until_label:
                    run_until_label = _coerce_str(run_until)
            if run_until_label:
                attributes["meta.run_until"] = run_until_label
            if pipeline_payload.phase:
                phase_label = _coerce_str(pipeline_payload.phase)
                if phase_label:
                    attributes["meta.phase"] = phase_label
                    attributes["meta.pipeline_phase"] = phase_label
    elif phase == "update_status":
        lifecycle_payload = transition.lifecycle
        if lifecycle_payload is not None:
            status = _coerce_str(lifecycle_payload.status)
            if status:
                attributes["meta.status"] = status
    elif phase == "ingest":
        embedding_section = transition.embedding
        if embedding_section is not None:
            embedding_result = getattr(embedding_section, "result", None)
            outcome = _coerce_str(getattr(embedding_result, "status", None))
            if outcome:
                attributes["meta.outcome"] = outcome
        delta_section = transition.delta
        if delta_section is not None:
            delta_decision = _coerce_str(delta_section.decision)
            if delta_decision:
                attributes.setdefault("meta.delta", delta_decision)
    elif phase == "guardrails":
        guardrail_payload = transition.guardrail
        if guardrail_payload is not None:
            allowed = getattr(guardrail_payload, "allowed", None)
            if isinstance(allowed, bool):
                attributes["meta.allowed"] = allowed
    elif phase == "ingest_decision":
        delta_section = transition.delta
        if delta_section is not None:
            delta_decision = _coerce_str(delta_section.decision)
            if delta_decision:
                attributes["meta.delta"] = delta_decision

    decision_value = _coerce_str(transition.decision)
    if phase in {"guardrails", "ingest_decision"} and decision_value:
        attributes["meta.decision"] = decision_value
    return attributes


def _ensure_ingestion_phase_spans(
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    trace_context: Mapping[str, Optional[str]],
) -> None:
    """Record fallback spans for ingestion phases when decorators are bypassed."""

    if not tracing_enabled():
        return

    transitions = result.get("transitions")
    if not isinstance(transitions, MappingABC):
        return

    recorded_phases: set[str] = set()
    span_tracker = state.get("_span_phases")
    if isinstance(span_tracker, (set, list, tuple)):
        recorded_phases.update(str(phase) for phase in span_tracker)

    graph_run_id = _coerce_str(result.get("graph_run_id")) or _coerce_str(
        state.get("graph_run_id")
    )
    base_metadata = _build_base_span_metadata(trace_context, graph_run_id)

    phase_mapping = {
        "update_status": "update_status_normalized",
        "guardrails": "enforce_guardrails",
        "document_pipeline": "document_pipeline",
        "ingest_decision": "ingest_decision",
        "ingest": "ingest",
    }

    for phase, transition_key in phase_mapping.items():
        if phase in recorded_phases:
            continue

        transition = _coerce_transition_result(transitions.get(transition_key))
        if transition is None:
            continue

        attributes = dict(base_metadata)
        attributes.update(_collect_transition_attributes(transition, phase))

        if attributes:
            record_span(f"crawler.ingestion.{phase}", attributes=attributes)


@shared_task(
    base=RetryableTask,
    queue="ingestion",
    accepts_scope=True,
    name="ai_core.tasks.run_ingestion_graph",
    time_limit=900,
    soft_time_limit=840,
)
def run_ingestion_graph(
    state: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the crawler ingestion LangGraph orchestration.

    Refactored from 127-line function with defensive glue to use:
    - IngestionContextBuilder: Extract metadata from nested dicts
    - ObservabilityWrapper: Manage tracing lifecycle
    - Cleaner orchestration flow
    """
    # 1. Build infrastructure components
    event_emitter = _resolve_event_emitter(meta)
    graph = _build_ingestion_graph(event_emitter)
    trace_context = _resolve_trace_context(state, meta)

    # 2. Extract ingestion context (defensive metadata extraction)
    context_builder = IngestionContextBuilder()
    ingestion_ctx = context_builder.build_from_state(
        state=state,
        meta=meta,
        trace_context=trace_context,
    )

    # 3. Normalize document input if needed
    # Note: Upload worker provides this, Crawler might not if legacy.
    # But for Universal Graph, we prefer normalized input.
    working_state = _prepare_working_state(
        state=state,
        ingestion_ctx=ingestion_ctx,
        trace_context=trace_context,
    )

    # 4. Setup observability (tracing)
    obs_wrapper = ObservabilityWrapper(observability_helpers)
    task_request = getattr(run_ingestion_graph, "request", None)
    obs_ctx = obs_wrapper.create_context(
        ingestion_ctx=ingestion_ctx,
        trace_context=trace_context,
        task_request=task_request,
    )

    obs_wrapper.start_trace(obs_ctx)

    try:
        # 5. Execute Universal Graph
        # Map legacy/worker state to UniversalIngestionInput

        # Determine source
        raw_source = ingestion_ctx.source or "upload"
        if raw_source.startswith("http://") or raw_source.startswith("https://"):
            source = "crawler"
        else:
            source = raw_source

        # Extract normalized document (it might be a dict or NormalizedDocument object)
        normalized_doc = working_state.get("normalized_document_input")
        if hasattr(normalized_doc, "model_dump"):
            normalized_doc = normalized_doc.model_dump(mode="json")

        # Extract collection_id
        collection_id = None
        if isinstance(normalized_doc, dict):
            ref = normalized_doc.get("ref", {})
            collection_id = ref.get("collection_id")

        if not collection_id:
            collection_id = ingestion_ctx.collection_id

        input_payload = {
            "source": source,
            "mode": "ingest_only",
            "collection_id": collection_id,
            "upload_blob": None,  # Provided via normalized_document for upload worker
            "metadata_obj": None,
            "normalized_document": normalized_doc,  # Key for Pre-normalized input
        }

        # BREAKING CHANGE (Option A): Business IDs extracted separately
        # They are NO LONGER part of ScopeContext after Phase 3 completion
        case_id = ingestion_ctx.case_id or "general"
        workflow_id = ingestion_ctx.workflow_id

        # Build Context via normalize_task_context (Pre-MVP ID Contract)
        # S2S Hop: service_id REQUIRED, user_id ABSENT
        # Note: case_id, workflow_id, collection_id parameters are DEPRECATED
        # but kept for backward compatibility
        scope = normalize_task_context(
            tenant_id=ingestion_ctx.tenant_id,
            case_id=case_id,  # DEPRECATED parameter (not in ScopeContext)
            service_id="celery-ingestion-worker",
            trace_id=ingestion_ctx.trace_id,
            invocation_id=getattr(ingestion_ctx, "invocation_id", None)
            or trace_context.get("invocation_id")
            or (meta.get("invocation_id") if meta else None),
            workflow_id=workflow_id,  # DEPRECATED parameter (not in ScopeContext)
            run_id=ingestion_ctx.run_id,
            ingestion_run_id=ingestion_ctx.ingestion_run_id,
            collection_id=collection_id,  # DEPRECATED parameter (not in ScopeContext)
        )

        # BREAKING CHANGE (Option A - Full ToolContext Migration):
        # Universal ingestion graph now expects nested ToolContext structure
        from ai_core.contracts.business import BusinessContext

        business = BusinessContext(
            case_id=case_id,
            workflow_id=workflow_id,
            collection_id=collection_id,
        )

        audit_meta = (meta or {}).get("audit_meta") if meta else {}
        run_context = {
            "scope": scope.model_dump(mode="json"),
            "business": business.model_dump(mode="json"),
            "metadata": {
                "dry_run": False,
                "audit_meta": dict(audit_meta or {}),
            },
        }

        result = graph.invoke({"input": input_payload, "context": run_context})

        # Output is in result["output"] usually, or result IS output?
        # Universal Graph returns UniversalIngestionOutput in 'output' key?
        # Wait, build_universal_ingestion_graph uses StateGraph.
        # invoke returns the final state.
        # UniversalIngestionState has 'output' key.
        final_output = result.get("output", {})

        # 6. Serialize result for Celery
        serialized_result = _jsonify_for_task(final_output)
        if not isinstance(serialized_result, dict):
            # Fallback if output structure is unexpected
            serialized_result = _jsonify_for_task(result)

        return serialized_result

    finally:
        try:
            obs_wrapper.end_trace()
        finally:
            # Cleanup temporary payload file
            if ingestion_ctx.raw_payload_path:
                from .ingestion import cleanup_raw_payload_artifact

                cleanup_raw_payload_artifact(ingestion_ctx.raw_payload_path)


def _prepare_working_state(
    state: Mapping[str, Any],
    ingestion_ctx,
    trace_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prepare working state with normalized document input.

    Extracted from run_ingestion_graph to isolate normalization logic.

    Args:
        state: Original graph state
        ingestion_ctx: Extracted ingestion context
        trace_context: Trace context

    Returns:
        Working state dict with normalized_document_input if applicable
    """
    working_state: Dict[str, Any] = dict(state)

    # Check if normalized input already present
    try:
        normalized_present = isinstance(
            working_state.get("normalized_document_input"), NormalizedDocument
        )
    except Exception:
        normalized_present = False

    if normalized_present:
        return working_state

    # Normalize from raw_document if available
    raw_reference = working_state.get("raw_document")
    if not isinstance(raw_reference, MappingABC):
        return working_state

    if not ingestion_ctx.tenant_id:
        return working_state

    try:
        contract = NormalizedDocumentInputV1.from_raw(
            raw_reference=raw_reference,
            tenant_id=ingestion_ctx.tenant_id,
            case_id=ingestion_ctx.case_id,
            workflow_id=ingestion_ctx.workflow_id,
            source=ingestion_ctx.source,
        )
        normalized_payload = normalize_from_raw(contract=contract)

        # Serialize to maintain JSON compatibility for Celery task payloads
        working_state["normalized_document_input"] = (
            normalized_payload.document.model_dump(mode="json")
        )

        try:
            working_state["document_id"] = str(
                normalized_payload.document.ref.document_id
            )
        except Exception:
            pass

    except Exception:
        # Fall back to original state; the graph will surface an error
        pass

    return working_state
