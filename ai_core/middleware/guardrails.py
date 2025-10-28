"""Shared guardrail limits and enforcement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

from ai_core.rag.guardrails import (
    FetcherLimits,
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)


class GuardrailStatus(str, Enum):
    """Guardrail evaluation result."""

    ALLOW = "allow"
    DENY = "deny"


class GuardrailErrorCategory(str, Enum):
    """Classification for guardrail violations used by error builders."""

    POLICY_DENY = "policy_deny"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class GuardrailDecision:
    """Decision emitted after evaluating guardrail limits."""

    decision: str
    reason: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        decision_value = str(self.decision or "").strip()
        if not decision_value:
            raise ValueError("decision_required")
        reason_value = str(self.reason or "").strip()
        if not reason_value:
            raise ValueError("reason_required")
        object.__setattr__(self, "decision", decision_value)
        object.__setattr__(self, "reason", reason_value)

        raw_attributes = self.attributes or {}
        if isinstance(raw_attributes, Mapping):
            proxy = MappingProxyType(dict(raw_attributes))
        else:
            raise TypeError("attributes_must_be_mapping")
        object.__setattr__(self, "attributes", proxy)

    @classmethod
    def from_legacy(
        cls,
        status: GuardrailStatus,
        reason: str,
        policy_events: Tuple[str, ...] = (),
        error: Optional[Any] = None,
    ) -> GuardrailDecision:
        attributes: dict[str, Any] = {"policy_events": tuple(policy_events)}
        if error is not None:
            attributes["error"] = error
        return cls(status.value, reason, attributes)

    @property
    def status(self) -> GuardrailStatus:
        return GuardrailStatus(self.decision)

    @property
    def policy_events(self) -> Tuple[str, ...]:
        events = self.attributes.get("policy_events", ())
        if isinstance(events, tuple):
            return events
        if isinstance(events, (list, set)):
            return tuple(events)
        return ()

    @property
    def error(self) -> Optional[Any]:
        return self.attributes.get("error")

    @property
    def allowed(self) -> bool:
        """Return ``True`` when the document may continue processing."""

        return self.status is GuardrailStatus.ALLOW


ErrorBuilder = Callable[
    [GuardrailErrorCategory, str, GuardrailSignals, Mapping[str, Any]],
    Optional[Any],
]


def enforce_guardrails(
    *,
    limits: Optional[GuardrailLimits] = None,
    signals: Optional[GuardrailSignals] = None,
    error_builder: Optional[ErrorBuilder] = None,
) -> GuardrailDecision:
    """Evaluate guardrails and emit an allow/deny decision."""

    applied_limits = limits or GuardrailLimits()
    applied_signals = signals or GuardrailSignals()

    host = applied_signals.host
    if host and host in applied_limits.host_blocklist:
        return _deny(
            "host_blocklisted",
            events=("host_blocklisted",),
            category=GuardrailErrorCategory.POLICY_DENY,
            error_builder=error_builder,
            signals=applied_signals,
            attributes={"host": host},
        )

    mime = applied_signals.mime_type
    if mime and mime in applied_limits.mime_blacklist:
        return _deny(
            "mime_blacklisted",
            events=("mime_blacklisted",),
            category=GuardrailErrorCategory.POLICY_DENY,
            error_builder=error_builder,
            signals=applied_signals,
            attributes={"mime_type": mime},
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
            category=GuardrailErrorCategory.POLICY_DENY,
            error_builder=error_builder,
            signals=applied_signals,
            attributes={"document_bytes": document_bytes, "limit_bytes": max_bytes},
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
            category=GuardrailErrorCategory.TIMEOUT,
            error_builder=error_builder,
            signals=applied_signals,
            attributes={
                "processing_time_ms": int(processing_time.total_seconds() * 1000),
                "limit_ms": int(processing_time_limit.total_seconds() * 1000),
            },
        )

    tenant_quota = applied_limits.tenant_quota
    tenant_usage = applied_signals.tenant_usage
    if tenant_quota is not None and tenant_usage is not None:
        if _quota_exceeded(tenant_quota, tenant_usage, document_bytes):
            return _deny(
                "tenant_quota_exceeded",
                events=("tenant_quota_exceeded",),
                category=GuardrailErrorCategory.POLICY_DENY,
                error_builder=error_builder,
                signals=applied_signals,
                attributes=_quota_attributes(
                    tenant_quota, tenant_usage, document_bytes
                ),
            )

    host_quota = applied_limits.host_quota
    host_usage = applied_signals.host_usage
    if host_quota is not None and host_usage is not None:
        if _quota_exceeded(host_quota, host_usage, document_bytes):
            return _deny(
                "host_quota_exceeded",
                events=("host_quota_exceeded",),
                category=GuardrailErrorCategory.POLICY_DENY,
                error_builder=error_builder,
                signals=applied_signals,
                attributes=_quota_attributes(host_quota, host_usage, document_bytes),
            )

    return GuardrailDecision.from_legacy(GuardrailStatus.ALLOW, "allow", (), None)


def _deny(
    reason: str,
    *,
    events: Tuple[str, ...],
    category: GuardrailErrorCategory,
    error_builder: Optional[ErrorBuilder],
    signals: GuardrailSignals,
    attributes: Mapping[str, Any],
) -> GuardrailDecision:
    error = _build_error(error_builder, category, reason, signals, attributes)
    return GuardrailDecision.from_legacy(GuardrailStatus.DENY, reason, events, error)


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
) -> dict[str, Any]:
    projected_documents = usage.documents + 1
    projected_bytes = usage.bytes + (document_bytes or 0)
    attributes: dict[str, Any] = {
        "projected_documents": projected_documents,
        "projected_bytes": projected_bytes,
    }
    if limits.max_documents is not None:
        attributes["limit_documents"] = limits.max_documents
    if limits.max_bytes is not None:
        attributes["limit_bytes"] = limits.max_bytes
    return attributes


def _build_error(
    builder: Optional[ErrorBuilder],
    category: GuardrailErrorCategory,
    reason: str,
    signals: GuardrailSignals,
    attributes: Mapping[str, Any],
) -> Optional[Any]:
    if builder is None:
        return None
    return builder(category, reason, signals, dict(attributes))


__all__ = [
    "ErrorBuilder",
    "FetcherLimits",
    "GuardrailDecision",
    "GuardrailErrorCategory",
    "GuardrailLimits",
    "GuardrailSignals",
    "GuardrailStatus",
    "QuotaLimits",
    "QuotaUsage",
    "enforce_guardrails",
]
