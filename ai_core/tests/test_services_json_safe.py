import json
from datetime import datetime, timezone
from uuid import uuid4

from ai_core import services
from ai_core.rag.vector_client import DedupSignatures, NearDuplicateSignature


def test_dump_jsonable_handles_dataclass_payloads():
    signatures = DedupSignatures(
        content_hash="9" * 64,
        near_duplicate=NearDuplicateSignature(
            fingerprint="fingerprint",
            tokens=(" Foo ", "bar", "foo"),
        ),
    )

    safe_payload = services._dump_jsonable({"signatures": signatures})

    assert safe_payload["signatures"]["content_hash"] == "9" * 64
    assert safe_payload["signatures"]["near_duplicate"]["fingerprint"] == "fingerprint"

    encoded = json.dumps(safe_payload)
    decoded = json.loads(encoded)
    assert decoded["signatures"]["near_duplicate"]["tokens"] == ["bar", "foo"]


def test_dump_jsonable_coerces_uuid_and_datetime():
    sample_id = uuid4()
    sample_time = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    safe_payload = services._dump_jsonable(
        {"document_id": sample_id, "created_at": sample_time}
    )

    assert safe_payload["document_id"] == str(sample_id)
    assert safe_payload["created_at"] == sample_time.isoformat().replace("+00:00", "Z")
    json.dumps(safe_payload)
