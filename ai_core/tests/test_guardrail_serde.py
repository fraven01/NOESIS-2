from datetime import timedelta

from common.guardrails import FetcherLimits
from crawler.fetcher import FetchFailure, FetchRequest, PolitenessContext

from ai_core.guardrails.serde import GuardrailSerde
from ai_core.rag.guardrails import (
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)


def test_guardrail_payload_roundtrip():
    limits = GuardrailLimits(
        max_document_bytes=2048,
        processing_time_limit=timedelta(seconds=30),
        mime_blacklist={"application/pdf"},
        tenant_quota=QuotaLimits(max_documents=5, max_bytes=1000),
    )
    signals = GuardrailSignals(
        tenant_id="tenant-a",
        provider="web",
        canonical_source="https://example.org/doc",
        host="example.org",
        document_bytes=512,
        processing_time=timedelta(seconds=2),
        mime_type="text/html",
        tenant_usage=QuotaUsage(documents=2, bytes=128),
    )

    payload = GuardrailSerde.to_payload(limits=limits, signals=signals)
    parsed = GuardrailSerde.from_payload(payload)

    assert parsed.limits == limits
    assert parsed.signals == signals
    assert parsed.config is None


def test_guardrail_from_payload_handles_config_mapping():
    config_payload = {"policy": "strict", "provider": "web"}
    parsed = GuardrailSerde.from_payload(config_payload)

    assert parsed.config == config_payload
    assert parsed.limits is None
    assert parsed.signals is None


def test_serialize_fetch_components():
    request = FetchRequest(
        canonical_source="https://example.org/data",
        politeness=PolitenessContext(host="example.org", slot="slot", crawl_delay=0.5),
        metadata={"foo": "bar"},
    )
    fetch_limits = FetcherLimits(max_bytes=4096, timeout=timedelta(seconds=5))
    failure = FetchFailure(reason="timeout", temporary=True)

    serialized_request = GuardrailSerde.serialize_fetch_request(request)
    serialized_limits = GuardrailSerde.serialize_fetch_limits(fetch_limits)
    failure_payload = GuardrailSerde.serialize_fetch_failure(failure)

    assert serialized_request["politeness"]["host"] == "example.org"
    assert serialized_limits["timeout_seconds"] == 5.0
    assert failure_payload == {"reason": "timeout", "temporary": True}
