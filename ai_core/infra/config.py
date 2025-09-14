from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


@lru_cache(maxsize=1)
def get_config() -> InfraConfig:
    """Load environment configuration.

    Values are read once and cached for subsequent calls.
    """

    return InfraConfig(
        litellm_base_url=env("LITELLM_BASE_URL"),
        litellm_api_key=env("LITELLM_API_KEY"),
        redis_url=env("REDIS_URL"),
        langfuse_public_key=env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=env("LANGFUSE_SECRET_KEY"),
    )
