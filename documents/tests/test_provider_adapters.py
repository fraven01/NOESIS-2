import pytest

from documents.contracts import DocumentMeta
from documents.providers import (
    ProviderReference,
    build_external_reference,
    parse_provider_reference,
)


def test_build_external_reference_normalizes_values() -> None:
    external_ref = build_external_reference(
        provider="  Web  ",
        external_id=" 1234 ",
        provider_tags={" collection ": "  Docs ", "": ""},
    )

    assert external_ref["provider"] == "Web"
    assert external_ref["external_id"] == "1234"
    assert external_ref["provider_tag:collection"] == "Docs"
    assert "provider_tag:" not in external_ref


def test_parse_provider_reference_roundtrip() -> None:
    external_ref = build_external_reference(
        provider="news",
        external_id="news::item-1",
        provider_tags={"collection": "daily"},
    )
    meta = DocumentMeta(
        tenant_id="tenant-1",
        workflow_id="wf-1",
        origin_uri="https://example.com/item",
        external_ref=external_ref,
    )

    provider = parse_provider_reference(meta)

    assert isinstance(provider, ProviderReference)
    assert provider.provider == "news"
    assert provider.external_id == "news::item-1"
    assert provider.canonical_source == "https://example.com/item"
    assert provider.provider_tags["collection"] == "daily"


def test_parse_provider_reference_handles_missing_tags() -> None:
    meta = DocumentMeta(
        tenant_id="tenant-1",
        workflow_id="wf-1",
        origin_uri="https://example.com/entry",
        external_ref={"provider": "source", "external_id": "id"},
    )

    provider = parse_provider_reference(meta)

    assert provider.provider == "source"
    assert provider.provider_tags == {}


def test_build_external_reference_rejects_empty_provider() -> None:
    with pytest.raises(ValueError):
        build_external_reference(provider="", external_id="id")
