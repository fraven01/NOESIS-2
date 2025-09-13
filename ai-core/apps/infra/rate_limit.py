"""Redis-backed token bucket rate limiter."""

from __future__ import annotations

import os

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
WINDOW = int(os.getenv("RATE_WINDOW", "60"))

_redis = redis.Redis.from_url(REDIS_URL)


def check(tenant: str) -> bool:
    """Return True if the tenant has remaining quota within the window."""
    key = f"rate:{tenant}"
    try:
        count = _redis.incr(key)
        if count == 1:
            _redis.expire(key, WINDOW)
        return count <= RATE_LIMIT
    except Exception:
        # If Redis is unavailable, do not block the request
        return True


def ready() -> bool:
    """Return True if Redis is reachable."""
    try:
        return bool(_redis.ping())
    except Exception:  # pragma: no cover - network
        return False
