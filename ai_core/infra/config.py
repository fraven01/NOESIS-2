from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Any

import environ


env = environ.Env()


@dataclass(frozen=True)
class InfraConfig:
    """Collected environment configuration for AI Core infrastructure."""

    litellm_base_url: str
    litellm_api_key: str
    redis_url: str
    langfuse_public_key: str
    langfuse_secret_key: str
    timeouts: dict[str, int]


def _parse_timeouts(raw: str | None) -> dict[str, int]:
    """Parse LiteLLM timeout mapping from an environment string."""

    if not raw:
        return {}

    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid LITELLM_TIMEOUTS JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("LITELLM_TIMEOUTS must be a JSON object")

    timeouts: dict[str, int] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("Timeout keys must be strings")
        try:
            timeout = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Timeout values must be integers"
            ) from exc
        if timeout < 0:
            raise ValueError("Timeout values must be non-negative")
        timeouts[key] = timeout

    return timeouts


@lru_cache(maxsize=1)
def get_config() -> InfraConfig:
    """Load environment configuration.

    Values are read once and cached for subsequent calls.
    """

    # Accept either LITELLM_API_KEY or fallback to LITELLM_MASTER_KEY for proxy auth
    api_key = env("LITELLM_API_KEY", default=None)
    if not api_key:
        api_key = env("LITELLM_MASTER_KEY", default="")

    return InfraConfig(
        litellm_base_url=env("LITELLM_BASE_URL"),
        litellm_api_key=api_key,
        redis_url=env("REDIS_URL"),
        langfuse_public_key=env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=env("LANGFUSE_SECRET_KEY"),
        timeouts=_parse_timeouts(env("LITELLM_TIMEOUTS", default=None)),
    )
