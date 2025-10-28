"""Guardrails enforcing crawler security and resource limits."""

from __future__ import annotations

from typing import Optional

from ai_core.middleware.guardrails import (
    GuardrailDecision,
    GuardrailErrorCategory,
    GuardrailStatus,
    enforce_guardrails as _shared_enforce_guardrails,
)
from ai_core.rag.guardrails import (
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)

from .errors import CrawlerError, ErrorClass


_ERROR_CLASS_BY_CATEGORY = {
    GuardrailErrorCategory.POLICY_DENY: ErrorClass.POLICY_DENY,
    GuardrailErrorCategory.TIMEOUT: ErrorClass.TIMEOUT,
}


def enforce_guardrails(
    *,
    limits: Optional[GuardrailLimits] = None,
    signals: Optional[GuardrailSignals] = None,
) -> GuardrailDecision:
    """Evaluate guardrails using the shared middleware implementation."""

    return _shared_enforce_guardrails(
        limits=limits,
        signals=signals,
        error_builder=_build_error,
    )


def _build_error(
    category: GuardrailErrorCategory,
    reason: str,
    signals: GuardrailSignals,
    attributes: dict,
) -> CrawlerError:
    error_class = _ERROR_CLASS_BY_CATEGORY.get(category, ErrorClass.POLICY_DENY)
    return CrawlerError(
        error_class,
        reason,
        source=signals.canonical_source,
        provider=signals.provider,
        attributes=attributes or {},
    )


__all__ = [
    "GuardrailDecision",
    "GuardrailLimits",
    "GuardrailSignals",
    "GuardrailStatus",
    "QuotaLimits",
    "QuotaUsage",
    "enforce_guardrails",
]
