"""Tests for collection contract models."""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from noesis_docs.contracts.collection_v1 import CollectionLink, CollectionRef


def test_collection_ref_full_payload_accepts_data() -> None:
    payload = {
        "tenant_id": " tenant-A ",
        "collection_id": uuid4(),
        "slug": " primary-collection ",
        "version_label": " v1 ",
    }

    model = CollectionRef(**payload)

    assert model.tenant_id == "tenant-A"
    assert model.slug == "primary-collection"
    assert model.version_label == "v1"
    assert model.collection_id == payload["collection_id"]
    assert isinstance(model.model_json_schema(), dict)


def test_collection_link_minimal_payload() -> None:
    payload = {
        "tenant_id": "tenant-B",
        "collection_id": uuid4(),
    }

    model = CollectionLink(**payload)

    assert model.slug is None
    assert model.version_label is None
    assert isinstance(model.model_json_schema(), dict)


@pytest.mark.parametrize("field_name", ["slug", "version_label"])
def test_collection_models_normalise_blank_optional_identifiers(field_name: str) -> None:
    payload = {
        "tenant_id": "tenant",
        "collection_id": uuid4(),
        field_name: "   ",
    }

    model = CollectionRef(**payload)

    assert getattr(model, field_name) is None


@pytest.mark.parametrize("tenant_id", ["", "   "])
def test_collection_ref_rejects_empty_tenant(tenant_id: str) -> None:
    with pytest.raises(ValidationError):
        CollectionRef(tenant_id=tenant_id, collection_id=uuid4())


def test_collection_ref_rejects_invalid_uuid() -> None:
    with pytest.raises(ValidationError):
        CollectionRef(tenant_id="tenant", collection_id="not-a-uuid")


@pytest.mark.parametrize("field_name", ["slug", "version_label"])
def test_collection_models_reject_overlong_identifiers(field_name: str) -> None:
    data = {
        "tenant_id": "tenant",
        "collection_id": uuid4(),
        field_name: "x" * 129,
    }

    with pytest.raises(ValidationError):
        CollectionRef(**data)

    with pytest.raises(ValidationError):
        CollectionLink(**data)


@pytest.mark.parametrize("field_name", ["slug", "version_label"])
def test_collection_models_reject_non_string_optional_identifiers(field_name: str) -> None:
    data = {
        "tenant_id": "tenant",
        "collection_id": uuid4(),
        field_name: 123,
    }

    with pytest.raises(TypeError):
        CollectionRef(**data)

    with pytest.raises(TypeError):
        CollectionLink(**data)
