"""Tests for the declarative crawler policy registry."""

from __future__ import annotations

import pytest

from crawler.contracts import (
    deregister_provider,
    get_provider_rules,
    normalize_source,
    register_provider,
)
from crawler.frontier import HostPolitenessPolicy, RecrawlFrequency
from crawler.policies import build_policy_registry


def test_policy_registry_overrides_provider_rules() -> None:
    original_rules = get_provider_rules("web")
    config = {
        "defaults": {
            "fetcher_limits": {"max_bytes": 1024},
            "recrawl_frequency": "standard",
        },
        "providers": {
            "web": {
                "param_whitelist": ["q"],
                "tracking_parameters": [],
            }
        },
    }

    registry = build_policy_registry(config)

    assert registry.default_fetcher_limits is not None
    assert registry.default_fetcher_limits.max_bytes == 1024
    assert registry.default_recrawl_frequency == RecrawlFrequency.STANDARD

    provider_policy = registry.get_provider_policy("web")
    assert provider_policy is not None
    assert provider_policy.rules.param_whitelist == frozenset({"q"})
    assert provider_policy.rules.tracking_parameters == frozenset()

    registry.apply_provider_rules()
    try:
        normalized = normalize_source(
            "web",
            "https://example.com/?q=1&ref=2&utm_source=x",
        )
        assert normalized.canonical_source == "https://example.com/?q=1"
    finally:
        register_provider(original_rules, replace=True)


def test_policy_registry_resolve_fetcher_limits_and_recrawl() -> None:
    config = {
        "defaults": {"fetcher_limits": {"max_bytes": 900}},
        "providers": {
            "web": {
                "fetcher_limits": {"max_bytes": 600},
                "recrawl_frequency": "rare",
            }
        },
        "hosts": {
            "example.com": {
                "fetcher_limits": {"max_bytes": 300},
                "recrawl_frequency": "frequent",
            }
        },
    }

    registry = build_policy_registry(config)

    assert registry.resolve_fetcher_limits("web").max_bytes == 600
    assert registry.resolve_fetcher_limits("web", host="example.com").max_bytes == 300
    assert registry.resolve_fetcher_limits("web", host="unknown.com").max_bytes == 600
    assert (
        registry.resolve_fetcher_limits("unknown", host="unknown.com").max_bytes == 900
    )

    assert (
        registry.resolve_recrawl_frequency("web", host="example.com")
        == RecrawlFrequency.FREQUENT
    )
    assert (
        registry.resolve_recrawl_frequency("web", host="other.com")
        == RecrawlFrequency.INFREQUENT
    )
    assert registry.resolve_recrawl_frequency("unknown", host="unknown.com") is None


def test_policy_registry_validates_host_providers() -> None:
    with pytest.raises(ValueError):
        build_policy_registry({"hosts": {"example.com": {"provider": "missing"}}})


def test_policy_registry_registers_new_provider() -> None:
    config = {
        "providers": {
            "api": {
                "canonicalizer": "http",
                "param_whitelist": ["token"],
            }
        },
        "hosts": {"api.example.com": {"provider": "api"}},
    }

    registry = build_policy_registry(config)
    provider_policy = registry.get_provider_policy("api")
    assert provider_policy is not None
    assert provider_policy.rules.param_whitelist == frozenset({"token"})

    registry.apply_provider_rules()
    try:
        normalized = normalize_source(
            "api",
            "https://api.example.com/resource?token=1&foo=2",
        )
        assert normalized.canonical_source == "https://api.example.com/resource?token=1"
    finally:
        deregister_provider("api")

    host_policy = registry.get_host_policy("api.example.com")
    assert host_policy is not None
    assert host_policy.provider_override == "api"


def test_policy_registry_politeness_resolution() -> None:
    config = {
        "hosts": {
            "news.example.com": {
                "politeness": {
                    "max_parallelism": 2,
                    "min_delay_seconds": 1.5,
                }
            }
        }
    }

    registry = build_policy_registry(config)
    politeness = registry.resolve_host_politeness("news.example.com")
    assert isinstance(politeness, HostPolitenessPolicy)
    assert politeness.max_parallelism == 2
    assert politeness.min_delay.total_seconds() == pytest.approx(1.5)


def test_policy_registry_rejects_invalid_mime_whitelist() -> None:
    config = {"defaults": {"fetcher_limits": {"mime_whitelist": []}}}
    with pytest.raises(ValueError):
        build_policy_registry(config)


def test_policy_registry_rejects_unknown_canonicalizer() -> None:
    config = {"providers": {"api": {"canonicalizer": "unknown"}}}
    with pytest.raises(ValueError):
        build_policy_registry(config)
