"""Tests for ``noesis_docs.contracts.collection_v1``."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from noesis_docs.contracts.collection_v1 import (
    CollectionLink,
    CollectionRef,
    collection_link_schema,
    collection_ref_schema,
)


def test_collection_ref_validates_and_normalises_fields() -> None:
    collection_uuid = uuid4()

    ref = CollectionRef(
        tenant_id=" tenant-A ",
        collection_id=str(collection_uuid),
        slug="  slug-1  ",
        version_label="  v1  ",
    )

    assert ref.collection_id == collection_uuid
    assert ref.tenant_id == "tenant-A"
    assert ref.slug == "slug-1"
    assert ref.version_label == "v1"


def test_collection_ref_optional_identifiers_blank_values_become_none() -> None:
    ref = CollectionRef(
        tenant_id="tenant",
        collection_id=uuid4(),
        slug="   ",
        version_label="\u200B",
    )

    assert ref.slug is None
    assert ref.version_label is None


@pytest.mark.parametrize("tenant_value", ["", "   ", "\u200B"])
def test_collection_ref_rejects_empty_tenant_id(tenant_value: str) -> None:
    with pytest.raises(ValidationError) as exc:
        CollectionRef(tenant_id=tenant_value, collection_id=uuid4())

    assert "tenant_id must not be empty" in str(exc.value)


def test_collection_ref_rejects_invalid_uuid() -> None:
    with pytest.raises(ValidationError) as exc:
        CollectionRef(tenant_id="tenant", collection_id="not-a-uuid")

    assert "collection_id" in str(exc.value)


def test_collection_ref_enforces_slug_length_limit() -> None:
    with pytest.raises(ValidationError) as exc:
        CollectionRef(
            tenant_id="tenant",
            collection_id=uuid4(),
            slug="a" * 129,
        )

    assert "slug" in str(exc.value)


def test_collection_ref_enforces_version_label_length_limit() -> None:
    with pytest.raises(ValidationError) as exc:
        CollectionRef(
            tenant_id="tenant",
            collection_id=uuid4(),
            version_label="b" * 129,
        )

    assert "version_label" in str(exc.value)


def test_collection_link_inherits_constraints() -> None:
    link = CollectionLink(tenant_id="tenant", collection_id=uuid4())

    assert isinstance(link, CollectionRef)


def test_schema_helpers_export_json_schema() -> None:
    ref_schema = collection_ref_schema()
    link_schema = collection_link_schema()

    assert ref_schema["title"] == "CollectionRef"
    assert ref_schema["properties"]["tenant_id"]["type"] == "string"
    assert link_schema["title"] == "CollectionLink"
    assert "collection_id" in link_schema["properties"]
