from __future__ import annotations

from datetime import timedelta

import pytest

from ai_core.middleware.guardrails import (
    GuardrailErrorCategory,
    enforce_guardrails as shared_enforce_guardrails,
)
from crawler.errors import ErrorClass
from crawler.guardrails import (
    GuardrailLimits,
    GuardrailSignals,
    GuardrailStatus,
    QuotaLimits,
    QuotaUsage,
    enforce_guardrails,
)


def test_allow_when_within_limits() -> None:
    limits = GuardrailLimits(
        max_document_bytes=1024,
        processing_time_limit=timedelta(seconds=5),
    )
    signals = GuardrailSignals(
        tenant_id="tenant",
        provider="web",
        canonical_source="https://example.com/doc",
        host="example.com",
        document_bytes=512,
        processing_time=timedelta(seconds=1),
        mime_type="text/html; charset=utf-8",
        tenant_usage=QuotaUsage(documents=0, bytes=0),
        host_usage=QuotaUsage(documents=0, bytes=0),
    )

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.ALLOW
    assert decision.allowed
    assert decision.policy_events == ()
    assert decision.error is None


def test_blocks_blocklisted_host() -> None:
    limits = GuardrailLimits(host_blocklist={"blocked.example"})
    signals = GuardrailSignals(host="Blocked.Example", provider="web")

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.reason == "host_blocklisted"
    assert decision.policy_events == ("host_blocklisted",)
    assert decision.error is not None
    assert decision.error.error_class is ErrorClass.POLICY_DENY
    assert decision.error.attributes["host"] == "blocked.example"


def test_blocks_blacklisted_mime() -> None:
    limits = GuardrailLimits(mime_blacklist={"application/x-msdownload"})
    signals = GuardrailSignals(mime_type="application/x-msdownload; q=0.8")

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.reason == "mime_blacklisted"
    assert decision.policy_events == ("mime_blacklisted",)
    assert decision.error is not None
    assert decision.error.error_class is ErrorClass.POLICY_DENY
    assert decision.error.attributes["mime_type"] == "application/x-msdownload"


def test_blocks_document_too_large() -> None:
    limits = GuardrailLimits(max_document_bytes=1024)
    signals = GuardrailSignals(document_bytes=4096)

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.reason == "document_too_large"
    assert decision.policy_events == ("max_document_bytes",)
    assert decision.error is not None
    assert decision.error.attributes == {
        "document_bytes": 4096,
        "limit_bytes": 1024,
    }


def test_blocks_processing_time_exceeded() -> None:
    limits = GuardrailLimits(processing_time_limit=timedelta(milliseconds=500))
    signals = GuardrailSignals(processing_time=timedelta(seconds=1))

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.reason == "processing_time_exceeded"
    assert decision.policy_events == ("processing_time_limit",)
    assert decision.error is not None
    assert decision.error.error_class is ErrorClass.TIMEOUT
    assert decision.error.attributes == {
        "processing_time_ms": 1000,
        "limit_ms": 500,
    }


def test_blocks_tenant_quota_documents() -> None:
    limits = GuardrailLimits(tenant_quota=QuotaLimits(max_documents=2))
    signals = GuardrailSignals(tenant_usage=QuotaUsage(documents=2))

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.reason == "tenant_quota_exceeded"
    assert decision.policy_events == ("tenant_quota_exceeded",)
    assert decision.error is not None
    assert decision.error.attributes["projected_documents"] == 3
    assert decision.error.attributes["limit_documents"] == 2


def test_blocks_host_quota_bytes() -> None:
    limits = GuardrailLimits(host_quota=QuotaLimits(max_bytes=10_000))
    signals = GuardrailSignals(
        host_usage=QuotaUsage(bytes=9_500),
        document_bytes=800,
    )

    decision = enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.reason == "host_quota_exceeded"
    assert decision.policy_events == ("host_quota_exceeded",)
    assert decision.error is not None
    assert decision.error.attributes["projected_bytes"] == 10_300
    assert decision.error.attributes["limit_bytes"] == 10_000


def test_invalid_quota_limits_raise() -> None:
    with pytest.raises(ValueError):
        GuardrailLimits(max_document_bytes=-1)
    with pytest.raises(ValueError):
        GuardrailLimits(processing_time_limit=timedelta(seconds=0))
    with pytest.raises(ValueError):
        QuotaLimits(max_documents=-5)
    with pytest.raises(ValueError):
        QuotaUsage(bytes=-1)
    with pytest.raises(ValueError):
        GuardrailSignals(document_bytes=-2)


def test_shared_guardrails_without_error_builder() -> None:
    limits = GuardrailLimits(max_document_bytes=10)
    signals = GuardrailSignals(document_bytes=20)

    decision = shared_enforce_guardrails(limits=limits, signals=signals)

    assert decision.status is GuardrailStatus.DENY
    assert decision.error is None
    assert decision.policy_events == ("max_document_bytes",)


def test_shared_guardrails_with_custom_builder() -> None:
    limits = GuardrailLimits(mime_blacklist={"application/json"})
    signals = GuardrailSignals(mime_type="application/json")

    captured: dict[str, object] = {}

    def _builder(category, reason, sig, attributes):  # type: ignore[no-untyped-def]
        captured["category"] = category
        captured["reason"] = reason
        captured["signals"] = sig
        captured["attributes"] = attributes
        return {"error": "custom"}

    decision = shared_enforce_guardrails(
        limits=limits,
        signals=signals,
        error_builder=_builder,
    )

    assert decision.error == {"error": "custom"}
    assert captured["category"] is GuardrailErrorCategory.POLICY_DENY
    assert captured["reason"] == "mime_blacklisted"
    assert isinstance(captured["signals"], GuardrailSignals)
    assert captured["attributes"] == {"mime_type": "application/json"}
