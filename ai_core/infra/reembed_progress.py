"""Helpers to track re-embedding progress in Redis."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Mapping

from redis import Redis
from redis.exceptions import RedisError, WatchError

from ai_core.infra.config import get_config
from common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PROGRESS_TTL_SECONDS = 7 * 24 * 60 * 60


def _redis_client() -> Redis | None:
    try:
        url = get_config().redis_url
    except Exception:
        url = None
    if not url:
        return None
    try:
        client = Redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except RedisError as exc:
        logger.warning("reembed.progress.redis_unavailable", extra={"error": str(exc)})
        return None


def init_reembed_progress(
    key: str,
    *,
    total_documents: int,
    total_chunks: int,
    embedding_profile: str,
    model_version: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    client = _redis_client()
    if client is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "status": "queued",
        "embedding_profile": embedding_profile,
        "model_version": model_version,
        "total_documents": int(total_documents),
        "total_chunks": int(total_chunks),
        "queued_documents": 0,
        "queued_chunks": 0,
        "processed_documents": 0,
        "processed_chunks": 0,
        "started_at": now,
        "updated_at": now,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    try:
        client.hset(key, mapping=payload)
        client.expire(key, int(DEFAULT_PROGRESS_TTL_SECONDS))
    except RedisError as exc:
        logger.warning("reembed.progress.init_failed", extra={"error": str(exc)})


def increment_reembed_progress(
    key: str,
    *,
    queued_documents: int = 0,
    queued_chunks: int = 0,
    processed_documents: int = 0,
    processed_chunks: int = 0,
) -> None:
    client = _redis_client()
    if client is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        pipe = client.pipeline()
        if queued_documents:
            pipe.hincrby(key, "queued_documents", int(queued_documents))
        if queued_chunks:
            pipe.hincrby(key, "queued_chunks", int(queued_chunks))
        if processed_documents:
            pipe.hincrby(key, "processed_documents", int(processed_documents))
        if processed_chunks:
            pipe.hincrby(key, "processed_chunks", int(processed_chunks))
        pipe.hset(key, "updated_at", now)
        pipe.execute()
    except RedisError as exc:
        logger.warning("reembed.progress.update_failed", extra={"error": str(exc)})


def reserve_reembed_delay(
    rate_key: str,
    *,
    chunk_count: int,
    quota_per_minute: int,
    ttl_seconds: int = 3600,
) -> float:
    """Reserve capacity in a per-tenant leaky bucket and return delay in seconds."""

    client = _redis_client()
    if client is None:
        return 0.0

    if quota_per_minute <= 0:
        return 0.0

    normalized_chunks = max(1, int(chunk_count))
    seconds_per_chunk = 60.0 / float(quota_per_minute)
    now = time.time()

    for _attempt in range(3):
        try:
            with client.pipeline() as pipe:
                pipe.watch(rate_key)
                current_raw = pipe.get(rate_key)
                next_ts = float(current_raw) if current_raw else now
                if next_ts < now:
                    next_ts = now
                delay = max(0.0, next_ts - now)
                new_next_ts = next_ts + (seconds_per_chunk * normalized_chunks)
                pipe.multi()
                pipe.set(rate_key, f"{new_next_ts:.6f}", ex=ttl_seconds)
                pipe.execute()
                return delay
        except WatchError:
            continue
        except RedisError as exc:
            logger.warning(
                "reembed.progress.rate_limit_failed",
                extra={"error": str(exc)},
            )
            return 0.0

    return 0.0
