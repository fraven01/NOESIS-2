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


def get_quota() -> int:
    """Return the per-tenant quota from ``AI_CORE_RATE_LIMIT_QUOTA``."""
    return env.int("AI_CORE_RATE_LIMIT_QUOTA", default=DEFAULT_QUOTA)


@lru_cache(maxsize=1)
def _get_redis() -> Redis:
    """Return a cached Redis client instance."""
    return Redis.from_url(get_config().redis_url, decode_responses=True)


def check(tenant: str, now: Optional[float] = None) -> bool:
    """Check whether the tenant has remaining requests in the current window.

    Parameters
    ----------
    tenant:
        Tenant identifier.
    now:
        Optional epoch timestamp used for testing.

    Returns
    -------
    bool
        ``True`` if the request is allowed, ``False`` otherwise.
    """

    quota = get_quota()
    ts = int(now if now is not None else time.time())
    window_start = ts - (ts % 60)
    ttl = 60 - (ts - window_start)
    key = f"rl:{tenant}"

    try:
        client = _get_redis()
        count = client.incr(key)
        if count == 1:
            client.expire(key, ttl)
        return count <= quota
    except RedisError as exc:
        logger.warning("rate limit fail-open for %s: %s", tenant, exc)
        return True
