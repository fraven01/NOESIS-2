"""Shared crawler error vocabulary for deterministic classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional


class ErrorClass(str, Enum):
    """Unified error classes emitted by crawler pipeline stages."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TRANSIENT_NETWORK = "transient_network"
    NOT_FOUND = "not_found"
    GONE = "gone"
    UNSUPPORTED_MEDIA = "unsupported_media"
    PARSER_FAILURE = "parser_failure"
    POLICY_DENY = "policy_deny"
    UPSTREAM_429 = "upstream_429"
    INGESTION_FAILURE = "ingestion_failure"


@dataclass(frozen=True)
class CrawlerError:
    """Structured error payload attached to crawler stage results."""

    error_class: ErrorClass
    reason: str
    source: Optional[str] = None
    provider: Optional[str] = None
    status_code: Optional[int] = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reason = (self.reason or "").strip()
        if not reason:
            raise ValueError("reason_required")
        object.__setattr__(self, "reason", reason)

        if self.source is not None:
            normalized_source = self.source.strip()
            object.__setattr__(self, "source", normalized_source or None)

        if self.provider is not None:
            normalized_provider = self.provider.strip()
            object.__setattr__(self, "provider", normalized_provider or None)

        if not isinstance(self.attributes, Mapping):
            raise TypeError("attributes_must_be_mapping")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
