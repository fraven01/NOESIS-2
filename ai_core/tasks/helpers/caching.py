from __future__ import annotations

from typing import Any, Mapping, Optional

from django.conf import settings
from redis import Redis

from ai_core.infra.config import get_config
from ai_core.infra.observability import emit_event
from common.logging import get_logger

from .embedding import _coerce_cache_part
from .task_utils import _task_context_payload

logger = get_logger(__name__)


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
