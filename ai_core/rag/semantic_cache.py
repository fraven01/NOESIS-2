from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai_core.rag.embeddings import EmbeddingClientError, get_embedding_client
from ai_core.tool_contracts import ToolContext
from common.logging import get_logger

logger = get_logger(__name__)


DEFAULT_TTL_S = 3600
DEFAULT_MAX_ITEMS = 200
DEFAULT_SIM_THRESHOLD = 0.9


@dataclass(frozen=True)
class CacheEntry:
    embedding: list[float]
    response: dict[str, Any]
    created_at: float
    last_access: float


@dataclass(frozen=True)
class CacheLookupResult:
    hit: bool
    response: Mapping[str, Any] | None
    embedding: list[float] | None
    similarity: float | None
    reason: str | None = None


def _parse_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value: object, *, fallback: int) -> int:
    try:
        candidate = int(str(value))
    except (TypeError, ValueError):
        return fallback
    if candidate <= 0:
        return fallback
    return candidate


def _coerce_float(value: object, *, fallback: float) -> float:
    try:
        candidate = float(str(value))
    except (TypeError, ValueError):
        return fallback
    if candidate <= 0.0:
        return fallback
    return candidate


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _scope_key(context: ToolContext) -> str:
    business = context.business
    return ":".join(
        [
            str(context.scope.tenant_id),
            str(business.case_id or "none"),
            str(business.collection_id or "none"),
            str(business.workflow_id or "none"),
        ]
    )


class SemanticCache:
    def __init__(
        self,
        *,
        enabled: bool,
        ttl_s: int,
        max_items: int,
        similarity_threshold: float,
    ) -> None:
        self._enabled = enabled
        self._ttl_s = ttl_s
        self._max_items = max_items
        self._similarity_threshold = similarity_threshold
        self._entries: dict[str, list[CacheEntry]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SemanticCache":
        enabled = _parse_bool(os.getenv("RAG_SEMANTIC_CACHE_ENABLED"), default=False)
        ttl_s = _coerce_int(
            os.getenv("RAG_SEMANTIC_CACHE_TTL_S"), fallback=DEFAULT_TTL_S
        )
        max_items = _coerce_int(
            os.getenv("RAG_SEMANTIC_CACHE_MAX_ITEMS"), fallback=DEFAULT_MAX_ITEMS
        )
        similarity_threshold = _coerce_float(
            os.getenv("RAG_SEMANTIC_CACHE_SIM_THRESHOLD"),
            fallback=DEFAULT_SIM_THRESHOLD,
        )
        return cls(
            enabled=enabled,
            ttl_s=ttl_s,
            max_items=max_items,
            similarity_threshold=similarity_threshold,
        )

    def lookup(self, query: str, context: ToolContext) -> CacheLookupResult:
        if not self._enabled:
            return CacheLookupResult(
                hit=False,
                response=None,
                embedding=None,
                similarity=None,
                reason="disabled",
            )
        query_text = (query or "").strip()
        if not query_text:
            return CacheLookupResult(
                hit=False,
                response=None,
                embedding=None,
                similarity=None,
                reason="empty_query",
            )
        embedding = self._embed_query(query_text)
        if embedding is None:
            return CacheLookupResult(
                hit=False,
                response=None,
                embedding=None,
                similarity=None,
                reason="embedding_failed",
            )

        scope = _scope_key(context)
        now = time.time()
        best_entry: CacheEntry | None = None
        best_similarity: float | None = None

        with self._lock:
            entries = self._entries.get(scope, [])
            kept: list[CacheEntry] = []
            for entry in entries:
                if now - entry.created_at > self._ttl_s:
                    continue
                kept.append(entry)
                similarity = _cosine_similarity(embedding, entry.embedding)
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
            if kept:
                self._entries[scope] = kept
            elif scope in self._entries:
                del self._entries[scope]

        if (
            best_entry is not None
            and best_similarity is not None
            and best_similarity >= self._similarity_threshold
        ):
            response = dict(best_entry.response)
            return CacheLookupResult(
                hit=True,
                response=response,
                embedding=embedding,
                similarity=best_similarity,
            )

        reason = "below_threshold"
        return CacheLookupResult(
            hit=False,
            response=None,
            embedding=embedding,
            similarity=best_similarity,
            reason=reason,
        )

    def store(
        self,
        query: str,
        context: ToolContext,
        response: Mapping[str, Any],
        *,
        embedding: Sequence[float] | None = None,
    ) -> None:
        if not self._enabled:
            return
        query_text = (query or "").strip()
        if not query_text:
            return
        if embedding is None:
            embedding = self._embed_query(query_text)
            if embedding is None:
                return
        payload = dict(response)
        now = time.time()
        entry = CacheEntry(
            embedding=list(embedding),
            response=payload,
            created_at=now,
            last_access=now,
        )
        scope = _scope_key(context)
        with self._lock:
            self._entries.setdefault(scope, []).append(entry)
            self._prune_locked(now)

    def _embed_query(self, query: str) -> list[float] | None:
        try:
            client = get_embedding_client()
            result = client.embed([query])
            if not result.vectors:
                return None
            return list(result.vectors[0])
        except EmbeddingClientError as exc:
            logger.warning(
                "rag.semantic_cache.embedding_failed",
                extra={"error": type(exc).__name__, "error_message": str(exc)},
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "rag.semantic_cache.embedding_failed",
                extra={"error": type(exc).__name__, "error_message": str(exc)},
            )
            return None

    def _prune_locked(self, now: float) -> None:
        for scope, entries in list(self._entries.items()):
            kept = [entry for entry in entries if now - entry.created_at <= self._ttl_s]
            if kept:
                self._entries[scope] = kept
            else:
                del self._entries[scope]

        total = sum(len(entries) for entries in self._entries.values())
        if total <= self._max_items:
            return
        flattened: list[tuple[float, str, CacheEntry]] = []
        for scope, entries in self._entries.items():
            for entry in entries:
                flattened.append((entry.created_at, scope, entry))
        flattened.sort(key=lambda item: item[0])
        remove_count = total - self._max_items
        for _, scope, entry in flattened[:remove_count]:
            entries = self._entries.get(scope, [])
            if entry in entries:
                entries.remove(entry)
            if not entries and scope in self._entries:
                del self._entries[scope]


_DEFAULT_CACHE: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = SemanticCache.from_env()
    return _DEFAULT_CACHE


def reset_semantic_cache() -> None:
    global _DEFAULT_CACHE
    _DEFAULT_CACHE = None


__all__ = [
    "CacheLookupResult",
    "SemanticCache",
    "get_semantic_cache",
    "reset_semantic_cache",
]
