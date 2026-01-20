from __future__ import annotations

import hashlib
import math
import os
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from django.conf import settings

from ai_core.infra import observability as observability_helpers
from ai_core.infra.observability import emit_event, tracing_enabled, update_observation
from ai_core.rag.embedding_config import (
    build_embedding_model_version,
    build_vector_space_id,
    get_embedding_profile,
)
from ai_core.rag.vector_space_resolver import resolve_vector_space_full
from common.logging import get_logger
from opentelemetry.trace import SpanKind

from .task_utils import _task_context_payload

logger = get_logger(__name__)

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
                candidate = tracer.start_as_current_span(
                    f"ingestion.embed.{name}", kind=SpanKind.INTERNAL
                )
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
