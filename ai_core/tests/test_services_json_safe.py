import json

from ai_core import services
from ai_core.rag.vector_client import DedupSignatures, NearDuplicateSignature


def test_make_json_safe_handles_dataclass_payloads():
    signatures = DedupSignatures(
        content_hash="9" * 64,
        near_duplicate=NearDuplicateSignature(
            fingerprint="fingerprint",
            tokens=(" Foo ", "bar", "foo"),
        ),
    )

    safe_payload = services._make_json_safe({"signatures": signatures})

    assert safe_payload["signatures"]["content_hash"] == "9" * 64
    assert safe_payload["signatures"]["near_duplicate"]["fingerprint"] == "fingerprint"

    encoded = json.dumps(safe_payload)
    decoded = json.loads(encoded)
    assert decoded["signatures"]["near_duplicate"]["tokens"] == ["bar", "foo"]
