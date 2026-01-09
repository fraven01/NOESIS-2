from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import environ
from redis import Redis
from redis.exceptions import RedisError

from .config import get_config
from common.logging import get_logger

env = environ.Env()
logger = get_logger(__name__)

DEFAULT_QUOTA = 60
DEFAULT_AGENTS_QUOTA = 100
DEFAULT_INGESTION_QUOTA = 500

_redis_client_cache: Optional[Redis] = None


def get_quota() -> int:
    """Return the per-tenant quota from ``AI_CORE_RATE_LIMIT_QUOTA``."""
    return env.int("AI_CORE_RATE_LIMIT_QUOTA", default=DEFAULT_QUOTA)


def get_agents_quota() -> int:
    """Return the per-tenant quota for agents tasks."""
    return env.int("AI_CORE_RATE_LIMIT_AGENTS_QUOTA", default=DEFAULT_AGENTS_QUOTA)


def get_ingestion_quota() -> int:
    """Return the per-tenant quota for ingestion tasks."""
    return env.int(
        "AI_CORE_RATE_LIMIT_INGESTION_QUOTA", default=DEFAULT_INGESTION_QUOTA
    )


def get_quota_for_scope(scope: str | None) -> int:
    """Return the quota for a logical rate limit scope."""
    if scope == "agents":
        return get_agents_quota()
    if scope == "ingestion":
        return get_ingestion_quota()
    return get_quota()


@lru_cache(maxsize=1)
def _get_redis() -> Redis:
    """Return a cached Redis client instance."""
    return Redis.from_url(get_config().redis_url, decode_responses=True)


_original_cache_clear = _get_redis.cache_clear


def _cache_clear_wrapper() -> None:
    global _redis_client_cache
    _redis_client_cache = None
    _original_cache_clear()


_get_redis.cache_clear = _cache_clear_wrapper


def _redis_client() -> Redis:
    """Return a module-level cached redis client."""

    global _redis_client_cache
    if _redis_client_cache is None:
        _redis_client_cache = _get_redis()
    return _redis_client_cache


def reset_cache() -> None:
    """Clear cached redis connections (testing helper)."""

    global _redis_client_cache
    _redis_client_cache = None
    try:
        _original_cache_clear()
    except Exception:
        pass


def check(tenant: str, now: Optional[float] = None) -> bool:
    """Check whether the tenant has remaining requests in the current window."""

    return check_scoped(tenant, scope=None, now=now, quota=get_quota())


def check_scoped(
    tenant: str,
    scope: str | None,
    now: Optional[float] = None,
    quota: Optional[int] = None,
) -> bool:
    """Check whether the tenant has remaining requests in a scoped window."""

    if quota is None:
        quota = get_quota_for_scope(scope)
    if quota is None or quota <= 0:
        return True

    ts = int(now if now is not None else time.time())
    window_start = ts - (ts % 60)
    ttl = 60 - (ts - window_start)
    prefix = f"rl:{scope}" if scope else "rl"
    key = f"{prefix}:{tenant}"

    try:
        client = _redis_client()
        count = client.incr(key)
        if count == 1:
            client.expire(key, ttl)
        return count <= quota
    except RedisError as exc:
        logger.warning("rate limit fail-open for %s: %s", tenant, exc)
        return True
