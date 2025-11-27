from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from documents.collection_service import (
    CollectionService,
    CollectionType,
    DEFAULT_MANUAL_COLLECTION_LABEL,
    DEFAULT_MANUAL_COLLECTION_SLUG,
)


class _StubDomainService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ensure_collection(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        result = SimpleNamespace()
        result.collection_id = kwargs.get("collection_id", uuid.uuid4())
        return result


def test_ensure_collection_injects_collection_type() -> None:
    domain = _StubDomainService()
    service = CollectionService(domain_service=domain)
    tenant = SimpleNamespace(id=uuid.uuid4())

    service.ensure_collection(
        tenant=tenant,
        key="alpha",
        name="Alpha",
        metadata={"channel": "crawler"},
        collection_type=CollectionType.SYSTEM,
    )

    assert len(domain.calls) == 1
    payload = domain.calls[0]
    assert payload["tenant"] is tenant
    assert payload["metadata"]["collection_type"] == CollectionType.SYSTEM
    assert payload["metadata"]["channel"] == "crawler"


def test_ensure_manual_collection_uses_domain_service(monkeypatch: pytest.MonkeyPatch):
    domain = _StubDomainService()
    service = CollectionService(domain_service=domain)
    tenant = SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(
        "documents.collection_service.TenantContext.resolve_identifier",
        lambda identifier, allow_pk=True: tenant,
    )

    result = service.ensure_manual_collection("tenant-id")

    assert result == str(domain.calls[0]["collection_id"])
    payload = domain.calls[0]
    assert payload["key"] == DEFAULT_MANUAL_COLLECTION_SLUG
    assert payload["name"] == DEFAULT_MANUAL_COLLECTION_LABEL
    assert payload["metadata"]["collection_type"] == CollectionType.SYSTEM
    assert payload["metadata"]["slug"] == DEFAULT_MANUAL_COLLECTION_SLUG
