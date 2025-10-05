"""Routing rules for embedding profile selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path

import yaml
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .embedding_config import get_embedding_configuration
from .selector_utils import normalise_selector_value


class RoutingConfigurationError(ImproperlyConfigured):
    """Raised when the embedding routing rules are invalid."""


class RoutingErrorCode:
    """Machine-readable error codes for routing configuration issues."""

    RULES_PATH_MISSING = "ROUTE_RULES_PATH_MISSING"
    RULES_FILE_MISSING = "ROUTE_RULES_FILE_MISSING"
    RULES_PARSE_FAILED = "ROUTE_RULES_PARSE_FAILED"
    ROOT_NOT_MAPPING = "ROUTE_RULES_ROOT_TYPE"
    FIELD_EMPTY = "ROUTE_FIELD_EMPTY"
    RULE_NOT_MAPPING = "ROUTE_RULE_TYPE"
    RULE_PROFILE_REQUIRED = "ROUTE_RULE_PROFILE_REQUIRED"
    DEFAULT_PROFILE_MISSING = "ROUTE_DEFAULT_PROFILE_MISSING"
    DEFAULT_PROFILE_EMPTY = "ROUTE_DEFAULT_PROFILE_EMPTY"
    RULES_NOT_SEQUENCE = "ROUTE_RULES_TYPE"
    UNKNOWN_PROFILE_DEFAULT = "ROUTE_UNKNOWN_PROFILE_DEFAULT"
    UNKNOWN_PROFILE_RULE = "ROUTE_UNKNOWN_PROFILE_RULE"
    DUPLICATE_SELECTOR = "ROUTE_DUP_SELECTOR"
    DUPLICATE_SELECTOR_SAME_TARGET = "ROUTE_DUP_SAME_TARGET"
    NO_MATCH = "ROUTE_NO_MATCH"
    CONFLICT = "ROUTE_CONFLICT"
    OVERLAP_SAME_SPECIFICITY = CONFLICT
    AMBIGUOUS_MATCH = CONFLICT


_ROUTING_DOC_HINT = (
    "Check config/rag_routing_rules.yaml and README.md (Fehlercodes Abschnitt) for details."
)


def _format_error(code: str, message: str) -> str:
    return f"{code}: {message}. {_ROUTING_DOC_HINT}"


def _format_selector(rule: RoutingRule) -> str:
    tenant = rule.tenant or "*"
    process = rule.process or "*"
    doc_class = rule.doc_class or "*"
    return f"tenant={tenant}, process={process}, doc_class={doc_class}"


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """A single routing rule describing an override for a profile."""

    profile: str
    tenant: str | None = None
    process: str | None = None
    doc_class: str | None = None

    def matches(
        self,
        *,
        tenant: str,
        process: str | None,
        doc_class: str | None,
    ) -> bool:
        if self.tenant is not None and tenant != self.tenant:
            return False
        if self.process is not None and process != self.process:
            return False
        if self.doc_class is not None and doc_class != self.doc_class:
            return False
        return True

    @property
    def specificity(self) -> int:
        return sum(
            value is not None for value in (self.tenant, self.process, self.doc_class)
        )


@dataclass(frozen=True, slots=True)
class RoutingTable:
    """Validated routing table for embedding profiles."""

    default_profile: str
    rules: tuple[RoutingRule, ...]

    def resolve(
        self,
        *,
        tenant: str,
        process: str | None,
        doc_class: str | None,
    ) -> str:
        best_rule: RoutingRule | None = None
        highest_specificity = -1

        for rule in self.rules:
            if not rule.matches(tenant=tenant, process=process, doc_class=doc_class):
                continue

            if rule.specificity > highest_specificity:
                best_rule = rule
                highest_specificity = rule.specificity
                continue

            if rule.specificity == highest_specificity and best_rule is not None:
                if rule.profile != best_rule.profile:
                    raise RoutingConfigurationError(
                        _format_error(
                            RoutingErrorCode.CONFLICT,
                            "Ambiguous routing rules match the same selector",
                        )
                    )

        if best_rule is not None:
            return best_rule.profile

        if not self.default_profile:
            raise RoutingConfigurationError(
                _format_error(
                    RoutingErrorCode.NO_MATCH,
                    "No routing rule or default profile available for selector",
                )
            )

        return self.default_profile


def _routing_rules_path() -> Path:
    raw_path = getattr(settings, "RAG_ROUTING_RULES_PATH", None)
    if raw_path is None:
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.RULES_PATH_MISSING,
                "RAG_ROUTING_RULES_PATH setting is missing",
            )
        )

    if isinstance(raw_path, Path):
        return raw_path

    path = Path(str(raw_path))
    return path


def _read_yaml(file_path: Path) -> Mapping[str, object]:
    if not file_path.exists():
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.RULES_FILE_MISSING,
                f"Routing rules file '{file_path}' does not exist",
            )
        )

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive parsing guard
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.RULES_PARSE_FAILED,
                "Failed to parse routing rules YAML",
            )
        ) from exc

    if not isinstance(data, Mapping):
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.ROOT_NOT_MAPPING,
                "Routing rules file must be a mapping",
            )
        )

    return data


def _normalise_optional(value: object | None, *, field: str) -> str | None:
    if value is None:
        return None

    normalized = normalise_selector_value(value)
    if normalized is None:
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.FIELD_EMPTY,
                f"{field} cannot be empty when provided",
            )
        )
    return normalized


def _build_rules(raw_rules: Sequence[object]) -> tuple[RoutingRule, ...]:
    rules: list[RoutingRule] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, Mapping):
            raise RoutingConfigurationError(
                _format_error(
                    RoutingErrorCode.RULE_NOT_MAPPING,
                    f"Routing rule #{index + 1} must be a mapping",
                )
            )

        profile = str(raw.get("profile", "")).strip()
        if not profile:
            raise RoutingConfigurationError(
                _format_error(
                    RoutingErrorCode.RULE_PROFILE_REQUIRED,
                    f"Routing rule #{index + 1} must declare a profile",
                )
            )

        rule = RoutingRule(
            profile=profile,
            tenant=_normalise_optional(raw.get("tenant"), field="tenant"),
            process=_normalise_optional(raw.get("process"), field="process"),
            doc_class=_normalise_optional(raw.get("doc_class"), field="doc_class"),
        )

        rules.append(rule)

    return tuple(rules)


def _ensure_profiles_exist(
    *,
    table: RoutingTable,
    available_profiles: Iterable[str],
) -> None:
    profiles = set(available_profiles)
    if table.default_profile not in profiles:
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.UNKNOWN_PROFILE_DEFAULT,
                f"Default routing profile '{table.default_profile}' is not configured",
            )
        )

    for rule in table.rules:
        if rule.profile not in profiles:
            raise RoutingConfigurationError(
                _format_error(
                    RoutingErrorCode.UNKNOWN_PROFILE_RULE,
                    f"Routing rule references unknown profile '{rule.profile}'",
                )
            )


def _validate_rule_uniqueness(rules: Sequence[RoutingRule]) -> None:
    seen_selectors: dict[tuple[str | None, str | None, str | None], RoutingRule] = {}
    duplicate_same_target: dict[tuple[str | None, str | None, str | None], RoutingRule] = {}
    unique_rules: list[RoutingRule] = []

    for rule in rules:
        selector = (rule.tenant, rule.process, rule.doc_class)
        if selector in seen_selectors:
            previous_rule = seen_selectors[selector]
            if previous_rule.profile != rule.profile:
                raise RoutingConfigurationError(
                    _format_error(
                        RoutingErrorCode.DUPLICATE_SELECTOR,
                        "Duplicate routing selector with different profile detected",
                    )
                )
            duplicate_same_target.setdefault(selector, previous_rule)
            continue

        seen_selectors[selector] = rule
        unique_rules.append(rule)

    for primary_rule in duplicate_same_target.values():
        LOGGER.warning(
            "%s: Duplicate routing selector for %s keeps profile '%s'; subsequent entries share the same target",
            RoutingErrorCode.DUPLICATE_SELECTOR_SAME_TARGET,
            _format_selector(primary_rule),
            primary_rule.profile,
        )

    for i, left in enumerate(unique_rules):
        for right in unique_rules[i + 1 :]:
            if left.specificity != right.specificity:
                continue

            if not _selectors_overlap(left, right):
                continue

            raise RoutingConfigurationError(
                _format_error(
                    RoutingErrorCode.CONFLICT,
                    "Overlapping routing rules with same specificity detected",
                )
            )


def _selectors_overlap(left: RoutingRule, right: RoutingRule) -> bool:
    if left.tenant is not None and right.tenant is not None and left.tenant != right.tenant:
        return False
    if (
        left.process is not None
        and right.process is not None
        and left.process != right.process
    ):
        return False
    if (
        left.doc_class is not None
        and right.doc_class is not None
        and left.doc_class != right.doc_class
    ):
        return False
    return True


@lru_cache(maxsize=1)
def get_routing_table() -> RoutingTable:
    """Load and validate the routing table from configuration."""

    raw_config = _read_yaml(_routing_rules_path())

    try:
        default_profile = str(raw_config["default_profile"]).strip()
    except KeyError as exc:
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.DEFAULT_PROFILE_MISSING,
                "default_profile is required",
            )
        ) from exc

    if not default_profile:
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.DEFAULT_PROFILE_EMPTY,
                "default_profile cannot be empty",
            )
        )

    raw_rules = raw_config.get("rules", [])
    if not isinstance(raw_rules, Sequence):
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.RULES_NOT_SEQUENCE,
                "rules must be a sequence of mappings",
            )
        )

    rules = _build_rules(raw_rules)

    table = RoutingTable(default_profile=default_profile, rules=rules)

    config = get_embedding_configuration()

    _ensure_profiles_exist(
        table=table,
        available_profiles=config.embedding_profiles.keys(),
    )
    _validate_rule_uniqueness(table.rules)

    return table


def resolve_embedding_profile_id(
    *, tenant: str, process: str | None = None, doc_class: str | None = None
) -> str:
    from .profile_resolver import resolve_embedding_profile

    return resolve_embedding_profile(
        tenant_id=tenant,
        process=process,
        doc_class=doc_class,
    )


def validate_routing_rules() -> None:
    """Validate routing rules at startup."""

    get_routing_table()


def reset_routing_rules_cache() -> None:
    """Clear cached routing rules (used in tests)."""

    get_routing_table.cache_clear()  # type: ignore[attr-defined]


__all__ = [
    "RoutingConfigurationError",
    "RoutingRule",
    "RoutingTable",
    "RoutingErrorCode",
    "get_routing_table",
    "resolve_embedding_profile_id",
    "reset_routing_rules_cache",
    "validate_routing_rules",
]
