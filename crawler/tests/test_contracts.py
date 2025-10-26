"""Tests for crawler canonicalization and external identifiers."""

import hashlib

import pytest

from crawler import (
    NormalizedSource,
    ProviderRules,
    deregister_provider,
    normalize_source,
    register_provider,
)
from crawler.contracts import MAX_EXTERNAL_ID_LENGTH, _default_http_canonicalizer


@pytest.fixture(scope="module", autouse=True)
def ensure_web_registered():
    """Ensure default web provider is available for tests."""

    # The module import registers the default provider; nothing to do but sanity-check.
    result = normalize_source("web", "https://example.com", None)
    assert isinstance(result, NormalizedSource)
    yield


def test_web_canonicalization_and_tracking_cleanup():
    first = normalize_source(
        "web",
        "https://Example.com:443/Folder/Page/?b=2&a=1&utm_source=newsletter#section",
    )
    second = normalize_source(
        "web",
        "https://example.com/Folder/Page/?a=1&b=2&utm_medium=email",
    )

    assert first.canonical_source == "https://example.com/Folder/Page/?a=1&b=2"
    assert second.canonical_source == first.canonical_source
    assert first.external_id == second.external_id
    assert first.provider_tags["source"] == first.canonical_source


def test_sub_identifier_extends_external_id():
    result = normalize_source(
        "web",
        "https://example.com/path/document.pdf",
        metadata={"item_id": "page-1"},
    )

    assert result.external_id == "web::https://example.com/path/document.pdf::page-1"
    assert result.provider_tags["item"] == "page-1"


def test_param_whitelist_filters_unknown_parameters():
    provider = "web-strict"
    rules = ProviderRules(
        name=provider,
        canonicalizer=_default_http_canonicalizer,
        param_whitelist=frozenset({"id"}),
    )
    register_provider(rules)
    try:
        result = normalize_source(
            provider,
            "https://example.com/resource?ID=42&preview=1&utm_source=ad",
        )
        assert result.canonical_source == "https://example.com/resource?ID=42"
    finally:
        deregister_provider(provider)


def test_native_id_provider_remains_stable():
    provider = "confluence"
    rules = ProviderRules(
        name=provider,
        canonicalizer=_default_http_canonicalizer,
        native_id_keys=("native_id", "page_id"),
    )
    register_provider(rules)
    try:
        meta = {"native_id": "123456", "collection": "space-A"}
        first = normalize_source(
            provider,
            "https://example.atlassian.net/wiki/spaces/SPACE/pages/123456?utm_campaign=test",
            metadata=meta,
        )
        second = normalize_source(
            provider,
            "https://example.atlassian.net/wiki/spaces/SPACE/pages/123456?ref=tracker",
            metadata=meta,
        )
        assert first.external_id == "confluence::123456"
        assert second.external_id == first.external_id
        assert first.provider_tags["native_id"] == "123456"
        assert first.provider_tags["collection"] == "space-A"
    finally:
        deregister_provider(provider)


def test_metadata_must_be_mapping():
    with pytest.raises(TypeError):
        normalize_source("web", "https://example.com", metadata=[("key", "value")])


def test_query_parameter_case_preserved():
    result = normalize_source(
        "web",
        "https://example.com/search?Foo=1&bar=2&foo=3",
    )

    assert result.canonical_source == "https://example.com/search?bar=2&Foo=1&foo=3"


def test_credentials_removed_from_canonical_source():
    result = normalize_source(
        "web",
        "https://user:pa55w0rd@example.com/private/data?id=5",
    )

    assert result.canonical_source == "https://example.com/private/data?id=5"
    assert result.external_id == "web::https://example.com/private/data?id=5"


def test_long_canonical_source_falls_back_to_hashed_external_id() -> None:
    long_segment = "a" * 1500
    source = f"https://example.com/resource/{long_segment}"

    result = normalize_source("web", source)

    expected_hash = hashlib.sha1(result.canonical_source.encode("utf-8")).hexdigest()
    assert result.external_id == f"web::sha1:{expected_hash}"
    assert len(result.external_id) <= MAX_EXTERNAL_ID_LENGTH
    assert result.provider_tags["source"] == result.canonical_source


def test_idna_host_canonicalization():
    result = normalize_source(
        "web",
        "https://www.müller.de/Überblick?Q=1",
    )

    expected = "https://www.xn--mller-kva.de/Überblick?Q=1"
    assert result.canonical_source == expected
    assert result.external_id == f"web::{expected}"


def test_default_ports_removed_from_canonical_source():
    http_result = normalize_source("web", "http://example.com:80/path")
    https_result = normalize_source("web", "https://example.com:443/path")

    assert http_result.canonical_source == "http://example.com/path"
    assert https_result.canonical_source == "https://example.com/path"


def test_fragment_strategy_preserve_and_empty():
    provider_preserve = "web-preserve"
    preserve_rules = ProviderRules(
        name=provider_preserve,
        canonicalizer=_default_http_canonicalizer,
        fragment_strategy="preserve",
    )
    provider_empty = "web-empty"
    empty_rules = ProviderRules(
        name=provider_empty,
        canonicalizer=_default_http_canonicalizer,
        fragment_strategy="empty",
    )
    register_provider(preserve_rules)
    register_provider(empty_rules)
    try:
        preserve = normalize_source(
            provider_preserve,
            "https://example.com/path?value=1#Section",
        )
        empty = normalize_source(
            provider_empty,
            "https://example.com/path?value=1#Section",
        )
        assert preserve.canonical_source == (
            "https://example.com/path?value=1#Section"
        )
        assert empty.canonical_source == "https://example.com/path?value=1#"
    finally:
        deregister_provider(provider_preserve)
        deregister_provider(provider_empty)


def test_query_parameter_sorting_is_stable_with_duplicates():
    messy = normalize_source(
        "web",
        "https://example.com/search?z=1&A=5&a=3&b=2&a=&B=1&empty=&z=&A=1",
    )
    reordered = normalize_source(
        "web",
        "https://example.com/search?A=1&b=2&empty=&A=5&z=&B=1&a=&a=3&z=1",
    )

    expected_query = "A=1&A=5&a=&a=3&B=1&b=2&empty=&z=&z=1"
    assert messy.canonical_source == f"https://example.com/search?{expected_query}"
    assert reordered.canonical_source == messy.canonical_source
    assert messy.external_id == reordered.external_id


def test_metadata_item_priority_respected():
    result = normalize_source(
        "web",
        "https://example.com/resource",
        metadata={"item_id": "meta-item", "document_key": "doc-key"},
    )

    assert result.provider_tags["item"] == "meta-item"


def test_normalize_source_rejects_unsupported_scheme():
    with pytest.raises(ValueError) as excinfo:
        normalize_source("web", "ftp://example.com/resource")

    assert str(excinfo.value) == "unsupported_scheme"
