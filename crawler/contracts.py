"""Canonicalization and external ID rules for crawler sources."""

from __future__ import annotations

import hashlib
import posixpath
import threading
import unicodedata
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MAX_EXTERNAL_ID_LENGTH = 512
_HASH_PREFIX = "sha1:"


@dataclass(frozen=True)
class Decision:
    """Canonical payload shared by crawler decision helpers."""

    decision: str
    reason: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_decision = str(self.decision or "").strip()
        if not normalized_decision:
            raise ValueError("decision_required")
        normalized_reason = str(self.reason or "").strip()
        if not normalized_reason:
            raise ValueError("reason_required")
        object.__setattr__(self, "decision", normalized_decision)
        object.__setattr__(self, "reason", normalized_reason)

        attributes: Mapping[str, Any]
        raw_attributes = self.attributes or {}
        if isinstance(raw_attributes, Mapping):
            attributes = MappingProxyType(dict(raw_attributes))
        else:
            raise TypeError("attributes_must_be_mapping")
        object.__setattr__(self, "attributes", attributes)


DEFAULT_TRACKING_PARAMETERS = frozenset(
    {
        "fbclid",
        "gclid",
        "igshid",
        "icid",
        "mc_eid",
        "msclkid",
        "ocid",
        "ref",
        "referrer",
        "ref_src",
        "spm",
        "utm_campaign",
        "utm_cid",
        "utm_content",
        "utm_id",
        "utm_medium",
        "utm_reader",
        "utm_source",
        "utm_term",
        "vero_conv",
        "vero_id",
    }
)

DEFAULT_TRACKING_PREFIXES = (
    "utm_",
    "mc_",
    "icid_",
    "mtm_",
    "pk_",
    "piwik_",
    "vero_",
)


@dataclass(frozen=True)
class ProviderRules:
    """Declarative rules describing canonicalization and identifiers."""

    name: str
    canonicalizer: Callable[[str, Optional[Mapping[str, Any]], "ProviderRules"], str]
    param_whitelist: Optional[frozenset[str]] = None
    tracking_parameters: frozenset[str] = DEFAULT_TRACKING_PARAMETERS
    tracking_prefixes: Tuple[str, ...] = DEFAULT_TRACKING_PREFIXES
    fragment_strategy: str = "drop"
    native_id_getter: Optional[
        Callable[[str, Optional[Mapping[str, Any]], "ProviderRules"], Optional[str]]
    ] = None
    native_id_keys: Tuple[str, ...] = ("native_id",)
    sub_identifier_keys: Tuple[str, ...] = (
        "item_id",
        "sub_id",
        "document_key",
        "page_id",
    )
    metadata_tag_keys: Mapping[str, str] = field(
        default_factory=lambda: {
            "collection": "collection",
            "item_id": "item",
            "sub_id": "item",
            "document_key": "item",
            "page_id": "item",
        }
    )

    def __post_init__(self) -> None:
        normalized_name = (self.name or "").strip().lower()
        if not normalized_name:
            raise ValueError("provider_name_empty")
        object.__setattr__(self, "name", normalized_name)
        if self.param_whitelist is not None:
            normalized = frozenset(name.lower() for name in self.param_whitelist)
            object.__setattr__(self, "param_whitelist", normalized)
        strategy = (self.fragment_strategy or "drop").lower()
        if strategy not in {"drop", "preserve", "empty"}:
            raise ValueError("fragment_strategy_invalid")
        object.__setattr__(self, "fragment_strategy", strategy)


class ProviderAlreadyRegisteredError(RuntimeError):
    """Raised when attempting to register a provider twice without replace."""


class ProviderNotRegisteredError(LookupError):
    """Raised when attempting to access an unknown provider."""


_PROVIDER_REGISTRY: Dict[str, ProviderRules] = {}
_PROVIDER_REGISTRY_LOCK = threading.RLock()


def register_provider(rules: ProviderRules, *, replace: bool = False) -> None:
    """Register provider rules in the in-memory registry."""

    key = rules.name
    with _PROVIDER_REGISTRY_LOCK:
        if not replace and key in _PROVIDER_REGISTRY:
            raise ProviderAlreadyRegisteredError(key)
        _PROVIDER_REGISTRY[key] = rules


def deregister_provider(name: str) -> None:
    """Remove provider rules from the registry."""

    with _PROVIDER_REGISTRY_LOCK:
        _PROVIDER_REGISTRY.pop(name, None)


def get_provider_rules(name: str) -> ProviderRules:
    """Return provider rules or raise :class:`ProviderNotRegisteredError`."""

    with _PROVIDER_REGISTRY_LOCK:
        try:
            return _PROVIDER_REGISTRY[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ProviderNotRegisteredError(name) from exc


@dataclass(frozen=True)
class NormalizedSource:
    """Canonical representation of a crawler source."""

    provider: str
    canonical_source: str
    external_id: str
    provider_tags: Mapping[str, str]


def _normalize_metadata_mapping(
    metadata: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return metadata
    raise TypeError("metadata must be a mapping")


def _normalize_tag_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
    else:
        candidate = str(value).strip()
    return candidate or None


def _default_http_canonicalizer(
    source: str, metadata: Optional[Mapping[str, Any]], rules: ProviderRules
) -> str:
    """Canonicalize HTTP/HTTPS URLs by normalizing case and query parameters."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    source = source.strip()
    if not source:
        raise ValueError("source_empty")

    parts = urlsplit(source)
    if not parts.scheme:
        raise ValueError("scheme_missing")
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("unsupported_scheme")

    raw_hostname = parts.hostname or ""
    if raw_hostname:
        normalized_hostname = unicodedata.normalize("NFC", raw_hostname)
    else:
        normalized_hostname = ""
    try:
        hostname = (
            normalized_hostname.encode("idna").decode("ascii")
            if normalized_hostname
            else ""
        )
    except UnicodeError:
        hostname = normalized_hostname
    hostname = hostname.lower()
    if not hostname:
        raise ValueError("host_missing")

    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    netloc = hostname
    if port:
        netloc = f"{hostname}:{port}"

    # Explicitly strip userinfo to avoid leaking credentials in canonical sources
    # or derived identifiers.

    path = _normalize_path(parts.path)
    query = _normalize_query(parts.query, rules)
    fragment = parts.fragment
    has_fragment_marker = "#" in source
    if rules.fragment_strategy == "drop":
        fragment = ""
        has_fragment_marker = False
    elif rules.fragment_strategy == "empty":
        fragment = ""

    canonical = urlunsplit((scheme, netloc, path, query, fragment))
    if rules.fragment_strategy == "empty" and has_fragment_marker:
        return f"{canonical}#"
    return canonical


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    normalized = posixpath.normpath(path)
    if path.endswith("/") and not normalized.endswith("/"):
        normalized = f"{normalized}/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized == "//":
        normalized = "/"
    return normalized


def _is_tracking_param(
    name: str, rules: ProviderRules, *, lower_name: Optional[str] = None
) -> bool:
    candidate = lower_name if lower_name is not None else name.lower()
    if candidate in rules.tracking_parameters:
        return True
    for prefix in rules.tracking_prefixes:
        if candidate.startswith(prefix):
            return True
    return False


def _normalize_query(query: str, rules: ProviderRules) -> str:
    if not query:
        return ""
    params = parse_qsl(query, keep_blank_values=True)
    filtered = []
    for name, value in params:
        normalized_name = name.strip()
        if not normalized_name:
            continue
        lower_name = normalized_name.lower()
        if _is_tracking_param(normalized_name, rules, lower_name=lower_name):
            continue
        if (
            rules.param_whitelist is not None
            and lower_name not in rules.param_whitelist
        ):
            continue
        filtered.append((lower_name, normalized_name, value))
    filtered.sort(key=lambda item: (item[0], item[1], item[2]))
    return urlencode(
        [(original_name, value) for _, original_name, value in filtered], doseq=True
    )


def _default_native_id_getter(
    source: str, metadata: Optional[Mapping[str, Any]], rules: ProviderRules
) -> Optional[str]:
    mapping = _normalize_metadata_mapping(metadata)
    for key in rules.native_id_keys:
        if key in mapping:
            value = _normalize_tag_value(mapping[key])
            if value:
                return value
    return None


def _extract_sub_identifier(
    mapping: Mapping[str, Any], rules: ProviderRules
) -> Optional[str]:
    for key in rules.sub_identifier_keys:
        if key in mapping:
            value = _normalize_tag_value(mapping[key])
            if value:
                return value
    return None


def _build_provider_tags(
    canonical_source: str,
    native_id: Optional[str],
    sub_identifier: Optional[str],
    mapping: Mapping[str, Any],
    rules: ProviderRules,
) -> Dict[str, str]:
    tags: Dict[str, str] = {"source": canonical_source}
    if native_id:
        tags["native_id"] = native_id
    if sub_identifier:
        tags["item"] = sub_identifier

    for metadata_key, tag_key in rules.metadata_tag_keys.items():
        if tag_key in tags:
            continue
        if metadata_key in mapping:
            value = _normalize_tag_value(mapping[metadata_key])
            if value:
                tags[tag_key] = value
    return tags


def normalize_source(
    provider: str,
    source: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> NormalizedSource:
    """Normalize a raw crawler source into canonical form and identifiers."""

    provider_key = provider.strip().lower()
    rules = get_provider_rules(provider_key)
    canonical_source = rules.canonicalizer(source, metadata, rules)

    mapping = _normalize_metadata_mapping(metadata)
    native_id = None
    if rules.native_id_getter is not None:
        native_id = rules.native_id_getter(source, mapping, rules)
    if native_id is None:
        native_id = _default_native_id_getter(source, mapping, rules)

    sub_identifier = _extract_sub_identifier(mapping, rules)

    base_identifier = native_id if native_id is not None else canonical_source
    external_id = _compose_external_id(provider_key, base_identifier, sub_identifier)

    provider_tags = _build_provider_tags(
        canonical_source, native_id, sub_identifier, mapping, rules
    )
    return NormalizedSource(
        provider=provider_key,
        canonical_source=canonical_source,
        external_id=external_id,
        provider_tags=provider_tags,
    )


# Export canonicalizer helpers for configuration registries.
HTTP_URL_CANONICALIZER = _default_http_canonicalizer
DEFAULT_NATIVE_ID_GETTER = _default_native_id_getter


def _compose_external_id(
    provider_key: str, base_identifier: str, sub_identifier: Optional[str]
) -> str:
    external_id = f"{provider_key}::{base_identifier}"
    if sub_identifier:
        external_id = f"{external_id}::{sub_identifier}"
    if len(external_id) <= MAX_EXTERNAL_ID_LENGTH:
        return external_id

    hashed_base = _hash_identifier(base_identifier)
    hashed_sub = _hash_identifier(sub_identifier) if sub_identifier else None

    fallback = f"{provider_key}::{hashed_base}"
    if hashed_sub:
        fallback = f"{fallback}::{hashed_sub}"
    return fallback


def _hash_identifier(identifier: str) -> str:
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


# Register default providers
register_provider(
    ProviderRules(
        name="web",
        canonicalizer=HTTP_URL_CANONICALIZER,
        param_whitelist=None,
        fragment_strategy="drop",
    )
)

