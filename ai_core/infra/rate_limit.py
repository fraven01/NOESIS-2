from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional

import environ
from redis import Redis
from redis.exceptions import RedisError

from .config import get_config

env = environ.Env()
logger = logging.getLogger(__name__)

DEFAULT_QUOTA = 60

_redis_client_cache: Optional[Redis] = None


def get_quota() -> int:
    """Return the per-tenant quota from ``AI_CORE_RATE_LIMIT_QUOTA``."""
    return env.int("AI_CORE_RATE_LIMIT_QUOTA", default=DEFAULT_QUOTA)


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

    quota = get_quota()
    ts = int(now if now is not None else time.time())
    window_start = ts - (ts % 60)
    ttl = 60 - (ts - window_start)
    key = f"rl:{tenant}"

    try:
        client = _redis_client()
        count = client.incr(key)
        if count == 1:
            client.expire(key, ttl)
        return count <= quota
    except RedisError as exc:
        logger.warning("rate limit fail-open for %s: %s", tenant, exc)
        return True
