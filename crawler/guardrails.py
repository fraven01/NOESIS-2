"""Guardrails enforcing crawler security and resource limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import FrozenSet, Optional, Tuple

from .errors import CrawlerError, ErrorClass
from .contracts import Decision


class GuardrailStatus(str, Enum):
    """Guardrail evaluation result."""

    ALLOW = "allow"
    DENY = "deny"


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
                _normalize_mime(mime)
                for mime in self.mime_blacklist
                if _normalize_mime(mime)
            ),
        )
        object.__setattr__(
            self,
            "host_blocklist",
            frozenset(
                _normalize_host(host)
                for host in self.host_blocklist
                if _normalize_host(host)
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


@dataclass(frozen=True)
class GuardrailDecision(Decision):
    """Decision emitted after evaluating guardrail limits."""

    @classmethod
    def from_legacy(
        cls,
        status: GuardrailStatus,
        reason: str,
        policy_events: Tuple[str, ...] = (),
        error: Optional[CrawlerError] = None,
    ) -> "GuardrailDecision":
        attributes = {"policy_events": tuple(policy_events)}
        if error is not None:
            attributes["error"] = error
        return cls(status.value, reason, attributes)

    @property
    def status(self) -> GuardrailStatus:
        return GuardrailStatus(self.decision)

    @property
    def policy_events(self) -> Tuple[str, ...]:
        return tuple(self.attributes.get("policy_events", ()))

    @property
    def error(self) -> Optional[CrawlerError]:
        return self.attributes.get("error")

    @property
    def allowed(self) -> bool:
        """Return ``True`` when the document may continue processing."""

        return self.status is GuardrailStatus.ALLOW


def enforce_guardrails(
    *,
    limits: Optional[GuardrailLimits] = None,
    signals: Optional[GuardrailSignals] = None,
) -> GuardrailDecision:
    """Evaluate guardrails and emit an allow/deny decision."""

    applied_limits = limits or GuardrailLimits()
    applied_signals = signals or GuardrailSignals()

    host = applied_signals.host
    if host and host in applied_limits.host_blocklist:
        return _deny(
            "host_blocklisted",
            events=("host_blocklisted",),
            error=_build_error(
                ErrorClass.POLICY_DENY,
                "host_blocklisted",
                applied_signals,
                attributes={"host": host},
            ),
        )

    mime = applied_signals.mime_type
    if mime and mime in applied_limits.mime_blacklist:
        return _deny(
            "mime_blacklisted",
            events=("mime_blacklisted",),
            error=_build_error(
                ErrorClass.POLICY_DENY,
                "mime_blacklisted",
                applied_signals,
                attributes={"mime_type": mime},
            ),
        )

    document_bytes = applied_signals.document_bytes
    max_bytes = applied_limits.max_document_bytes
    if (
        max_bytes is not None
        and document_bytes is not None
        and document_bytes > max_bytes
    ):
        return _deny(
            "document_too_large",
            events=("max_document_bytes",),
            error=_build_error(
                ErrorClass.POLICY_DENY,
                "document_too_large",
                applied_signals,
                attributes={"document_bytes": document_bytes, "limit_bytes": max_bytes},
            ),
        )

    processing_time_limit = applied_limits.processing_time_limit
    processing_time = applied_signals.processing_time
    if (
        processing_time_limit is not None
        and processing_time is not None
        and processing_time > processing_time_limit
    ):
        return _deny(
            "processing_time_exceeded",
            events=("processing_time_limit",),
            error=_build_error(
                ErrorClass.TIMEOUT,
                "processing_time_exceeded",
                applied_signals,
                attributes={
                    "processing_time_ms": int(processing_time.total_seconds() * 1000),
                    "limit_ms": int(processing_time_limit.total_seconds() * 1000),
                },
            ),
        )

    tenant_quota = applied_limits.tenant_quota
    tenant_usage = applied_signals.tenant_usage
    if tenant_quota is not None and tenant_usage is not None:
        if _quota_exceeded(tenant_quota, tenant_usage, document_bytes):
            return _deny(
                "tenant_quota_exceeded",
                events=("tenant_quota_exceeded",),
                error=_build_error(
                    ErrorClass.POLICY_DENY,
                    "tenant_quota_exceeded",
                    applied_signals,
                    attributes=_quota_attributes(
                        tenant_quota, tenant_usage, document_bytes
                    ),
                ),
            )

    host_quota = applied_limits.host_quota
    host_usage = applied_signals.host_usage
    if host_quota is not None and host_usage is not None:
        if _quota_exceeded(host_quota, host_usage, document_bytes):
            return _deny(
                "host_quota_exceeded",
                events=("host_quota_exceeded",),
                error=_build_error(
                    ErrorClass.POLICY_DENY,
                    "host_quota_exceeded",
                    applied_signals,
                    attributes=_quota_attributes(
                        host_quota, host_usage, document_bytes
                    ),
                ),
            )

    return GuardrailDecision.from_legacy(
        GuardrailStatus.ALLOW, "allow", (), None
    )


def _deny(
    reason: str,
    *,
    events: Tuple[str, ...],
    error: CrawlerError,
) -> GuardrailDecision:
    return GuardrailDecision.from_legacy(
        GuardrailStatus.DENY, reason, events, error
    )


def _quota_exceeded(
    limits: QuotaLimits, usage: QuotaUsage, document_bytes: Optional[int]
) -> bool:
    projected_documents = usage.documents + 1
    if limits.max_documents is not None and projected_documents > limits.max_documents:
        return True
    projected_bytes = usage.bytes + (document_bytes or 0)
    if limits.max_bytes is not None and projected_bytes > limits.max_bytes:
        return True
    return False


def _quota_attributes(
    limits: QuotaLimits, usage: QuotaUsage, document_bytes: Optional[int]
) -> dict:
    projected_documents = usage.documents + 1
    projected_bytes = usage.bytes + (document_bytes or 0)
    attributes = {
        "projected_documents": projected_documents,
        "projected_bytes": projected_bytes,
    }
    if limits.max_documents is not None:
        attributes["limit_documents"] = limits.max_documents
    if limits.max_bytes is not None:
        attributes["limit_bytes"] = limits.max_bytes
    return attributes


def _build_error(
    error_class: ErrorClass,
    reason: str,
    signals: GuardrailSignals,
    *,
    attributes: Optional[dict] = None,
) -> CrawlerError:
    return CrawlerError(
        error_class,
        reason,
        source=signals.canonical_source,
        provider=signals.provider,
        attributes=attributes or {},
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
