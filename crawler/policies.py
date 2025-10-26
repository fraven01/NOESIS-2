"""Declarative configuration registry for crawler policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Callable, Collection, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import (
    DEFAULT_NATIVE_ID_GETTER,
    HTTP_URL_CANONICALIZER,
    ProviderNotRegisteredError,
    ProviderRules,
    get_provider_rules,
    register_provider,
)
from .fetcher import FetcherLimits
from .frontier import HostPolitenessPolicy, RecrawlFrequency

Canonicalizer = Callable[[str, Optional[Mapping[str, Any]], ProviderRules], str]
NativeIdGetter = Callable[
    [str, Optional[Mapping[str, Any]], ProviderRules], Optional[str]
]

_UNSET = object()

_DEFAULT_CANONICALIZERS: Dict[str, Canonicalizer] = {"http": HTTP_URL_CANONICALIZER}
_DEFAULT_NATIVE_ID_GETTERS: Dict[str, NativeIdGetter] = {
    "default": DEFAULT_NATIVE_ID_GETTER,
}


@dataclass(frozen=True)
class ProviderPolicy:
    """Materialized provider configuration and associated limits."""

    name: str
    rules: ProviderRules
    fetcher_limits: Optional[FetcherLimits] = None
    recrawl_frequency: Optional[RecrawlFrequency] = None


@dataclass(frozen=True)
class HostPolicy:
    """Host-level overrides for fetch, recrawl, and politeness policies."""

    host: str
    provider_override: Optional[str] = None
    fetcher_limits: Optional[FetcherLimits] = None
    recrawl_frequency: Optional[RecrawlFrequency] = None
    politeness: Optional[HostPolitenessPolicy] = None


@dataclass(frozen=True)
class PolicyRegistry:
    """Immutable view over resolved crawler policies.

    The registry is designed for hot-reload scenarios by instantiating a new
    object and swapping references atomically. Call :meth:`apply_provider_rules`
    to register provider rules with the canonicalization layer when deploying
    a freshly constructed registry. Mutating an instance in place is not
    thread-safe and may leave processes with stale rules; build a replacement
    registry instead.
    """

    providers: Mapping[str, ProviderPolicy]
    hosts: Mapping[str, HostPolicy]
    default_fetcher_limits: Optional[FetcherLimits] = None
    default_recrawl_frequency: Optional[RecrawlFrequency] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))
        object.__setattr__(self, "hosts", MappingProxyType(dict(self.hosts)))

    def get_provider_policy(self, provider: Optional[str]) -> Optional[ProviderPolicy]:
        """Return the policy for ``provider`` if configured."""

        if not provider:
            return None
        key = provider.strip().lower()
        if not key:
            return None
        return self.providers.get(key)

    def get_host_policy(self, host: Optional[str]) -> Optional[HostPolicy]:
        """Return the host-specific policy if defined."""

        if not host:
            return None
        key = host.strip().lower()
        if not key:
            return None
        return self.hosts.get(key)

    def resolve_fetcher_limits(
        self,
        provider: Optional[str],
        *,
        host: Optional[str] = None,
    ) -> Optional[FetcherLimits]:
        """Return the most specific fetcher limits for ``host``/``provider``."""

        host_policy = self.get_host_policy(host)
        if host_policy and host_policy.fetcher_limits is not None:
            return host_policy.fetcher_limits
        provider_policy = self.get_provider_policy(provider)
        if provider_policy and provider_policy.fetcher_limits is not None:
            return provider_policy.fetcher_limits
        return self.default_fetcher_limits

    def resolve_recrawl_frequency(
        self,
        provider: Optional[str],
        *,
        host: Optional[str] = None,
    ) -> Optional[RecrawlFrequency]:
        """Return the recrawl bucket honoring host overrides."""

        host_policy = self.get_host_policy(host)
        if host_policy and host_policy.recrawl_frequency is not None:
            return host_policy.recrawl_frequency
        provider_policy = self.get_provider_policy(provider)
        if provider_policy and provider_policy.recrawl_frequency is not None:
            return provider_policy.recrawl_frequency
        return self.default_recrawl_frequency

    def resolve_host_politeness(
        self, host: Optional[str]
    ) -> Optional[HostPolitenessPolicy]:
        """Return the host politeness override if configured."""

        host_policy = self.get_host_policy(host)
        if host_policy:
            return host_policy.politeness
        return None

    def resolve_provider(
        self, host: Optional[str], default_provider: Optional[str] = None
    ) -> Optional[str]:
        """Return the provider override for ``host`` if present."""

        host_policy = self.get_host_policy(host)
        if host_policy and host_policy.provider_override is not None:
            return host_policy.provider_override
        return default_provider

    def apply_provider_rules(self, *, replace: bool = True) -> None:
        """Register provider rules contained in the registry."""

        for policy in self.providers.values():
            register_provider(policy.rules, replace=replace)


def build_policy_registry(
    config: Mapping[str, Any],
    *,
    canonicalizers: Optional[Mapping[str, Canonicalizer]] = None,
    native_id_getters: Optional[Mapping[str, NativeIdGetter]] = None,
) -> PolicyRegistry:
    """Build a :class:`PolicyRegistry` from a declarative configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("config_must_be_mapping")

    canonicalizer_registry = _prepare_callable_registry(
        canonicalizers, _DEFAULT_CANONICALIZERS, "canonicalizers"
    )
    native_id_registry = _prepare_callable_registry(
        native_id_getters, _DEFAULT_NATIVE_ID_GETTERS, "native_id_getters"
    )

    defaults_cfg = _ensure_mapping(config.get("defaults"), "defaults")

    default_fetcher_limits = _parse_fetcher_limits(
        defaults_cfg.get("fetcher_limits", _UNSET), "defaults.fetcher_limits"
    )
    default_recrawl = _parse_recrawl_frequency(
        defaults_cfg.get("recrawl_frequency", _UNSET), "defaults.recrawl_frequency"
    )

    providers_cfg = _ensure_mapping(config.get("providers"), "providers")
    provider_keys = {
        _normalize_registry_key(name, "providers") for name in providers_cfg.keys()
    }
    providers: Dict[str, ProviderPolicy] = {}
    for raw_name, provider_cfg in providers_cfg.items():
        provider_name = _normalize_registry_key(raw_name, "providers")
        mapping = _ensure_mapping(provider_cfg, f"providers[{raw_name}]")
        providers[provider_name] = _build_provider_policy(
            provider_name,
            mapping,
            canonicalizer_registry,
            native_id_registry,
        )

    hosts_cfg = _ensure_mapping(config.get("hosts"), "hosts")
    hosts: Dict[str, HostPolicy] = {}
    for raw_host, host_cfg in hosts_cfg.items():
        host_name = _normalize_registry_key(raw_host, "hosts")
        mapping = _ensure_mapping(host_cfg, f"hosts[{raw_host}]")
        hosts[host_name] = _build_host_policy(
            host_name,
            mapping,
            provider_keys,
        )

    return PolicyRegistry(
        providers=providers,
        hosts=hosts,
        default_fetcher_limits=default_fetcher_limits,
        default_recrawl_frequency=default_recrawl,
    )


def _prepare_callable_registry(
    overrides: Optional[Mapping[str, Callable]],
    defaults: Mapping[str, Callable],
    context: str,
) -> Dict[str, Callable]:
    registry = {key: value for key, value in defaults.items()}
    if overrides is None:
        return registry
    if not isinstance(overrides, Mapping):
        raise TypeError(f"{context}_must_be_mapping")
    for key, value in overrides.items():
        if not callable(value):
            raise TypeError(f"{context}[{key!r}]_must_be_callable")
        registry[_normalize_registry_key(key, context)] = value
    return registry


def _build_provider_policy(
    name: str,
    config: Mapping[str, Any],
    canonicalizers: Mapping[str, Canonicalizer],
    native_id_getters: Mapping[str, NativeIdGetter],
) -> ProviderPolicy:
    context = f"providers[{name}]"
    allowed_keys = {
        "copy_from",
        "canonicalizer",
        "native_id_getter",
        "param_whitelist",
        "tracking_parameters",
        "tracking_prefixes",
        "fragment_strategy",
        "native_id_keys",
        "sub_identifier_keys",
        "metadata_tag_keys",
        "fetcher_limits",
        "recrawl_frequency",
    }
    _validate_keys(config, allowed_keys, context)

    base_rules: Optional[ProviderRules] = None
    copy_from_value = config.get("copy_from")
    if copy_from_value is not None:
        copy_from = _normalize_registry_key(copy_from_value, f"{context}.copy_from")
        try:
            base_rules = get_provider_rules(copy_from)
        except ProviderNotRegisteredError as exc:
            raise ValueError(f"{context}.copy_from_unknown:{copy_from}") from exc
    else:
        try:
            base_rules = get_provider_rules(name)
        except ProviderNotRegisteredError:
            base_rules = None

    canonicalizer_override = config.get("canonicalizer", _UNSET)
    if canonicalizer_override is not _UNSET:
        canonicalizer = _lookup_canonicalizer(
            canonicalizer_override, canonicalizers, f"{context}.canonicalizer"
        )
    elif base_rules is not None:
        canonicalizer = base_rules.canonicalizer
    else:
        canonicalizer = _lookup_canonicalizer(
            "http", canonicalizers, f"{context}.canonicalizer"
        )

    if base_rules is None:
        base_rules = ProviderRules(name=name, canonicalizer=canonicalizer)
    else:
        base_rules = replace(base_rules, name=name, canonicalizer=canonicalizer)

    native_id_override = config.get("native_id_getter", _UNSET)
    if native_id_override is not _UNSET:
        native_id_getter = _lookup_native_id_getter(
            native_id_override, native_id_getters, f"{context}.native_id_getter"
        )
    else:
        native_id_getter = base_rules.native_id_getter

    param_whitelist_value = config.get("param_whitelist", _UNSET)
    if param_whitelist_value is not _UNSET:
        param_whitelist = _parse_param_whitelist(
            param_whitelist_value, f"{context}.param_whitelist"
        )
    else:
        param_whitelist = base_rules.param_whitelist

    tracking_parameters_value = config.get("tracking_parameters", _UNSET)
    if tracking_parameters_value is not _UNSET:
        tracking_parameters = _parse_tracking_parameters(
            tracking_parameters_value, f"{context}.tracking_parameters"
        )
    else:
        tracking_parameters = base_rules.tracking_parameters

    tracking_prefixes_value = config.get("tracking_prefixes", _UNSET)
    if tracking_prefixes_value is not _UNSET:
        tracking_prefixes = _parse_tracking_prefixes(
            tracking_prefixes_value, f"{context}.tracking_prefixes"
        )
    else:
        tracking_prefixes = base_rules.tracking_prefixes

    fragment_strategy_value = config.get("fragment_strategy", _UNSET)
    if fragment_strategy_value is not _UNSET:
        fragment_strategy = _parse_fragment_strategy(
            fragment_strategy_value, f"{context}.fragment_strategy"
        )
    else:
        fragment_strategy = base_rules.fragment_strategy

    native_id_keys_value = config.get("native_id_keys", _UNSET)
    if native_id_keys_value is not _UNSET:
        native_id_keys = _parse_identifier_keys(
            native_id_keys_value, f"{context}.native_id_keys"
        )
    else:
        native_id_keys = base_rules.native_id_keys

    sub_identifier_keys_value = config.get("sub_identifier_keys", _UNSET)
    if sub_identifier_keys_value is not _UNSET:
        sub_identifier_keys = _parse_identifier_keys(
            sub_identifier_keys_value, f"{context}.sub_identifier_keys"
        )
    else:
        sub_identifier_keys = base_rules.sub_identifier_keys

    metadata_tag_keys_value = config.get("metadata_tag_keys", _UNSET)
    if metadata_tag_keys_value is not _UNSET:
        metadata_tag_keys = dict(base_rules.metadata_tag_keys)
        metadata_tag_keys.update(
            _parse_metadata_tag_keys(
                metadata_tag_keys_value, f"{context}.metadata_tag_keys"
            )
        )
    else:
        metadata_tag_keys = base_rules.metadata_tag_keys

    updated_rules = replace(
        base_rules,
        param_whitelist=param_whitelist,
        tracking_parameters=tracking_parameters,
        tracking_prefixes=tracking_prefixes,
        fragment_strategy=fragment_strategy,
        native_id_getter=native_id_getter,
        native_id_keys=native_id_keys,
        sub_identifier_keys=sub_identifier_keys,
        metadata_tag_keys=metadata_tag_keys,
    )

    fetcher_limits = _parse_fetcher_limits(
        config.get("fetcher_limits", _UNSET), f"{context}.fetcher_limits"
    )
    recrawl_frequency = _parse_recrawl_frequency(
        config.get("recrawl_frequency", _UNSET), f"{context}.recrawl_frequency"
    )

    return ProviderPolicy(
        name=name,
        rules=updated_rules,
        fetcher_limits=fetcher_limits,
        recrawl_frequency=recrawl_frequency,
    )


def _build_host_policy(
    host: str,
    config: Mapping[str, Any],
    configured_providers: Collection[str],
) -> HostPolicy:
    context = f"hosts[{host}]"
    allowed_keys = {"provider", "fetcher_limits", "recrawl_frequency", "politeness"}
    _validate_keys(config, allowed_keys, context)

    provider_value = config.get("provider")
    provider_override: Optional[str] = None
    if provider_value is not None:
        provider_override = _normalize_registry_key(
            provider_value, f"{context}.provider"
        )
        if provider_override not in configured_providers:
            try:
                get_provider_rules(provider_override)
            except ProviderNotRegisteredError as exc:
                raise ValueError(
                    f"{context}.provider_unknown:{provider_override}"
                ) from exc

    fetcher_limits = _parse_fetcher_limits(
        config.get("fetcher_limits", _UNSET), f"{context}.fetcher_limits"
    )
    recrawl_frequency = _parse_recrawl_frequency(
        config.get("recrawl_frequency", _UNSET), f"{context}.recrawl_frequency"
    )
    politeness = _parse_politeness(
        config.get("politeness", _UNSET), f"{context}.politeness"
    )

    return HostPolicy(
        host=host,
        provider_override=provider_override,
        fetcher_limits=fetcher_limits,
        recrawl_frequency=recrawl_frequency,
        politeness=politeness,
    )


def _normalize_registry_key(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context}_must_be_string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{context}_empty")
    return normalized


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}_must_be_mapping")
    return value


def _validate_keys(
    config: Mapping[str, Any], allowed: Sequence[str], context: str
) -> None:
    invalid = sorted(set(config.keys()) - set(allowed))
    if invalid:
        raise ValueError(f"{context}.unknown_keys:{','.join(invalid)}")


def _lookup_canonicalizer(
    value: Any, registry: Mapping[str, Canonicalizer], context: str
) -> Canonicalizer:
    if callable(value):
        return value  # type: ignore[return-value]
    key = _normalize_registry_key(value, context)
    try:
        return registry[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{context}_unknown:{key}") from exc


def _lookup_native_id_getter(
    value: Any, registry: Mapping[str, NativeIdGetter], context: str
) -> NativeIdGetter:
    if callable(value):
        return value  # type: ignore[return-value]
    key = _normalize_registry_key(value, context)
    try:
        return registry[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{context}_unknown:{key}") from exc


def _parse_param_whitelist(value: Any, context: str) -> Optional[frozenset[str]]:
    if value is None:
        return None
    items = _parse_string_sequence(value, context, lower=True)
    return frozenset(items)


def _parse_tracking_parameters(value: Any, context: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    items = _parse_string_sequence(value, context, lower=True)
    return frozenset(items)


def _parse_tracking_prefixes(value: Any, context: str) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    items = _parse_string_sequence(value, context, lower=True)
    return tuple(items)


def _parse_fragment_strategy(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context}_must_be_string")
    normalized = value.strip().lower()
    if normalized not in {"drop", "preserve", "empty"}:
        raise ValueError(f"{context}_invalid:{value}")
    return normalized


def _parse_identifier_keys(value: Any, context: str) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    items = _parse_string_sequence(value, context, lower=False)
    return tuple(items)


def _parse_metadata_tag_keys(value: Any, context: str) -> Dict[str, str]:
    mapping = _ensure_mapping(value, context)
    normalized: Dict[str, str] = {}
    for key, tag in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{context}_invalid_key")
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"{context}_invalid_value")
        normalized[key.strip()] = tag.strip()
    return normalized


def _parse_fetcher_limits(value: Any, context: str) -> Optional[FetcherLimits]:
    if value is _UNSET or value is None:
        return None
    mapping = _ensure_mapping(value, context)
    allowed_keys = {"max_bytes", "timeout_seconds", "mime_whitelist"}
    _validate_keys(mapping, allowed_keys, context)

    kwargs: Dict[str, Any] = {}
    has_value = False
    if "max_bytes" in mapping:
        max_bytes = mapping["max_bytes"]
        if not isinstance(max_bytes, int):
            raise TypeError(f"{context}.max_bytes_must_be_int")
        if max_bytes <= 0:
            raise ValueError(f"{context}.max_bytes_positive")
        kwargs["max_bytes"] = max_bytes
        has_value = True

    if "timeout_seconds" in mapping:
        timeout_seconds = mapping["timeout_seconds"]
        if not isinstance(timeout_seconds, (int, float)):
            raise TypeError(f"{context}.timeout_seconds_must_be_number")
        if timeout_seconds <= 0:
            raise ValueError(f"{context}.timeout_seconds_positive")
        kwargs["timeout"] = timedelta(seconds=float(timeout_seconds))
        has_value = True

    if "mime_whitelist" in mapping:
        mime_value = mapping["mime_whitelist"]
        if not isinstance(mime_value, Sequence) or isinstance(mime_value, str):
            raise TypeError(f"{context}.mime_whitelist_sequence")
        mime_items = []
        for entry in mime_value:
            if not isinstance(entry, str):
                raise TypeError(f"{context}.mime_whitelist_string")
            candidate = entry.strip().lower()
            if not candidate:
                raise ValueError(f"{context}.mime_whitelist_empty_entry")
            mime_items.append(candidate)
        if not mime_items:
            raise ValueError(f"{context}.mime_whitelist_empty")
        kwargs["mime_whitelist"] = tuple(mime_items)
        has_value = True

    if not has_value:
        return None
    return FetcherLimits(**kwargs)


def _parse_recrawl_frequency(value: Any, context: str) -> Optional[RecrawlFrequency]:
    if value is _UNSET or value is None:
        return None
    if isinstance(value, RecrawlFrequency):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{context}_must_be_string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{context}_empty")
    if normalized == "infrequent":
        normalized = RecrawlFrequency.INFREQUENT.value
    try:
        return RecrawlFrequency(normalized)
    except ValueError as exc:
        raise ValueError(f"{context}_invalid:{value}") from exc


def _parse_politeness(value: Any, context: str) -> Optional[HostPolitenessPolicy]:
    if value is _UNSET or value is None:
        return None
    mapping = _ensure_mapping(value, context)
    allowed_keys = {"max_parallelism", "min_delay_seconds"}
    _validate_keys(mapping, allowed_keys, context)

    kwargs: Dict[str, Any] = {}
    if "max_parallelism" in mapping:
        max_parallelism = mapping["max_parallelism"]
        if not isinstance(max_parallelism, int):
            raise TypeError(f"{context}.max_parallelism_must_be_int")
        kwargs["max_parallelism"] = max_parallelism

    if "min_delay_seconds" in mapping:
        min_delay = mapping["min_delay_seconds"]
        if not isinstance(min_delay, (int, float)):
            raise TypeError(f"{context}.min_delay_seconds_must_be_number")
        if min_delay < 0:
            raise ValueError(f"{context}.min_delay_seconds_non_negative")
        kwargs["min_delay"] = timedelta(seconds=float(min_delay))

    return HostPolitenessPolicy(**kwargs)


def _parse_string_sequence(value: Any, context: str, *, lower: bool) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{context}_must_be_sequence")
    items = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(f"{context}_entries_must_be_string")
        candidate = entry.strip()
        if not candidate:
            raise ValueError(f"{context}_empty_entry")
        items.append(candidate.lower() if lower else candidate)
    return tuple(items)


__all__ = [
    "ProviderPolicy",
    "HostPolicy",
    "PolicyRegistry",
    "build_policy_registry",
]
