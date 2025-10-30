"""Guardrail limit structures shared across crawler and ingestion flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import FrozenSet, Optional

from common.guardrails import FetcherLimits


@dataclass(frozen=True)
class QuotaLimits:
    """Configurable quota thresholds for tenants or hosts."""

    max_documents: Optional[int] = None
    max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_documents is not None and self.max_documents < 0:
            raise ValueError("max_documents_negative")
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ValueError("max_bytes_negative")


@dataclass(frozen=True)
class QuotaUsage:
    """Observed quota consumption before admitting the current document."""

    documents: int = 0
    bytes: int = 0

    def __post_init__(self) -> None:
        if self.documents < 0:
            raise ValueError("documents_negative")
        if self.bytes < 0:
            raise ValueError("bytes_negative")


@dataclass(frozen=True)
class GuardrailLimits:
    """Top-level guardrails for crawler document processing."""

    max_document_bytes: Optional[int] = None
    processing_time_limit: Optional[timedelta] = None
    mime_blacklist: FrozenSet[str] = frozenset()
    host_blocklist: FrozenSet[str] = frozenset()
    tenant_quota: Optional[QuotaLimits] = None
    host_quota: Optional[QuotaLimits] = None

    def __post_init__(self) -> None:
        if self.max_document_bytes is not None and self.max_document_bytes < 0:
            raise ValueError("max_document_bytes_negative")
        if (
            self.processing_time_limit is not None
            and self.processing_time_limit <= timedelta(0)
        ):
            raise ValueError("processing_time_limit_non_positive")
        object.__setattr__(
            self,
            "mime_blacklist",
            frozenset(
                entry
                for entry in (_normalize_mime(mime) for mime in self.mime_blacklist)
                if entry is not None
            ),
        )
        object.__setattr__(
            self,
            "host_blocklist",
            frozenset(
                entry
                for entry in (_normalize_host(host) for host in self.host_blocklist)
                if entry is not None
            ),
        )


@dataclass(frozen=True)
class GuardrailSignals:
    """Signals describing the document under evaluation."""

    tenant_id: Optional[str] = None
    provider: Optional[str] = None
    canonical_source: Optional[str] = None
    host: Optional[str] = None
    document_bytes: Optional[int] = None
    processing_time: Optional[timedelta] = None
    mime_type: Optional[str] = None
    tenant_usage: Optional[QuotaUsage] = None
    host_usage: Optional[QuotaUsage] = None

    def __post_init__(self) -> None:
        if self.document_bytes is not None and self.document_bytes < 0:
            raise ValueError("document_bytes_negative")
        if self.processing_time is not None and self.processing_time < timedelta(0):
            raise ValueError("processing_time_negative")
        if self.mime_type is not None:
            object.__setattr__(self, "mime_type", _normalize_mime(self.mime_type))
        if self.host is not None:
            object.__setattr__(self, "host", _normalize_host(self.host))
        if self.tenant_id is not None:
            object.__setattr__(self, "tenant_id", self.tenant_id.strip() or None)
        if self.provider is not None:
            object.__setattr__(self, "provider", self.provider.strip() or None)
        if self.canonical_source is not None:
            object.__setattr__(
                self, "canonical_source", self.canonical_source.strip() or None
            )


def _normalize_mime(mime: Optional[str]) -> Optional[str]:
    if mime is None:
        return None
    normalized = mime.split(";", 1)[0].strip().lower()
    return normalized or None


def _normalize_host(host: Optional[str]) -> Optional[str]:
    if host is None:
        return None
    normalized = host.strip().lower()
    return normalized or None


__all__ = [
    "FetcherLimits",
    "GuardrailLimits",
    "GuardrailSignals",
    "QuotaLimits",
    "QuotaUsage",
]
