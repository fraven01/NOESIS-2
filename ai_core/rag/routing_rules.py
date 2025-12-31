"""Routing rules for embedding profile selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any

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


_ROUTING_DOC_HINT = "Check config/rag_routing_rules.yaml and README.md (Fehlercodes Abschnitt) for details."


def _format_error(code: str, message: str) -> str:
    return f"{code}: {message}. {_ROUTING_DOC_HINT}"


def _format_selector(rule: RoutingRule) -> str:
    tenant = rule.tenant or "*"
    process = rule.process or "*"
    workflow_id = rule.workflow_id or "*"
    collection_id = rule.collection_id or "*"
    doc_class = rule.doc_class or "*"
    return (
        f"tenant={tenant}, "
        f"process={process}, "
        f"workflow_id={workflow_id}, "
        f"collection_id={collection_id}, "
        f"doc_class={doc_class}"
    )


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """A single routing rule describing an override for a profile and/or chunker mode."""

    profile: str
    index: int
    tenant: str | None = None
    process: str | None = None
    collection_id: str | None = None
    workflow_id: str | None = None
    doc_class: str | None = None
    chunker_mode: str | None = None  # late | agentic | hybrid

    def matches(
        self,
        *,
        tenant: str,
        process: str | None,
        collection_id: str | None,
        workflow_id: str | None,
        doc_class: str | None,
    ) -> bool:
        if self.tenant is not None and tenant != self.tenant:
            return False
        if self.collection_id is not None and collection_id != self.collection_id:
            return False
        if self.workflow_id is not None and workflow_id != self.workflow_id:
            return False
        if self.process is not None and process != self.process:
            return False
        if self.doc_class is not None and doc_class != self.doc_class:
            return False
        return True

    @property
    def specificity(self) -> int:
        return sum(
            value is not None
            for value in (
                self.tenant,
                self.process,
                self.collection_id,
                self.workflow_id,
                self.doc_class,
            )
        )

    @property
    def priority_key(self) -> tuple[int, int, int, int, int, int]:
        tenant_flag = int(self.tenant is not None)
        process_flag = int(self.process is not None)
        collection_flag = int(self.collection_id is not None)
        workflow_flag = int(self.workflow_id is not None)
        doc_class_flag = int(self.doc_class is not None)
        selector_count = self.specificity
        return (
            collection_flag,
            workflow_flag,
            process_flag,
            doc_class_flag,
            tenant_flag,
            selector_count,
        )

    @property
    def selector_tuple(
        self,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        return (
            self.tenant,
            self.process,
            self.collection_id,
            self.workflow_id,
            self.doc_class,
        )


@dataclass(frozen=True, slots=True)
class RoutingResolution:
    """Result metadata for routing table lookups."""

    profile: str
    rule: RoutingRule | None

    @property
    def resolver_path(self) -> str:
        if self.rule is None:
            return "default_profile"
        return f"rules[{self.rule.index}]"

    @property
    def fallback_used(self) -> bool:
        return self.rule is None


@dataclass(frozen=True, slots=True)
class RoutingTable:
    """Validated routing table for embedding profiles and chunker modes."""

    default_profile: str
    rules: tuple[RoutingRule, ...]
    default_chunker_mode: str = "late"  # late | agentic | hybrid

    def resolve(
        self,
        *,
        tenant: str,
        process: str | None,
        collection_id: str | None,
        workflow_id: str | None,
        doc_class: str | None,
    ) -> str:
        return self._resolve_internal(
            tenant=tenant,
            process=process,
            collection_id=collection_id,
            workflow_id=workflow_id,
            doc_class=doc_class,
        ).profile

    def resolve_with_metadata(
        self,
        *,
        tenant: str,
        process: str | None,
        collection_id: str | None,
        workflow_id: str | None,
        doc_class: str | None,
    ) -> RoutingResolution:
        return self._resolve_internal(
            tenant=tenant,
            process=process,
            collection_id=collection_id,
            workflow_id=workflow_id,
            doc_class=doc_class,
        )

    def _resolve_internal(
        self,
        *,
        tenant: str,
        process: str | None,
        collection_id: str | None,
        workflow_id: str | None,
        doc_class: str | None,
    ) -> RoutingResolution:
        best_rule: RoutingRule | None = None
        best_priority: tuple[int, int, int, int, int, int] | None = None

        for rule in self.rules:
            if not rule.matches(
                tenant=tenant,
                process=process,
                collection_id=collection_id,
                workflow_id=workflow_id,
                doc_class=doc_class,
            ):
                continue

            priority = rule.priority_key

            if best_priority is None or priority > best_priority:
                best_rule = rule
                best_priority = priority
                continue

            if priority == best_priority and best_rule is not None:
                if rule.profile != best_rule.profile:
                    raise RoutingConfigurationError(
                        _format_error(
                            RoutingErrorCode.CONFLICT,
                            "Ambiguous routing rules match the same selector",
                        )
                    )

        if best_rule is not None:
            return RoutingResolution(profile=best_rule.profile, rule=best_rule)

        if not self.default_profile:
            raise RoutingConfigurationError(
                _format_error(
                    RoutingErrorCode.NO_MATCH,
                    "No routing rule or default profile available for selector",
                )
            )

        return RoutingResolution(profile=self.default_profile, rule=None)


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


def _use_collection_routing() -> bool:
    flags: Mapping[str, Any] | None = getattr(settings, "RAG_ROUTING_FLAGS", None)
    if isinstance(flags, Mapping):
        flag_value = flags.get("rag.use_collection_routing")
        if isinstance(flag_value, bool):
            return flag_value
        if isinstance(flag_value, str):
            lowered = flag_value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False

    env_override = os.getenv("RAG_USE_COLLECTION_ROUTING")
    if env_override is not None:
        return env_override.strip().lower() in {"1", "true", "yes", "on"}

    return False


def is_collection_routing_enabled() -> bool:
    """Return whether collection-based routing is enabled."""

    return _use_collection_routing()


def _build_rules(raw_rules: Sequence[object]) -> tuple[RoutingRule, ...]:
    rules: list[RoutingRule] = []
    use_collection_routing = _use_collection_routing()
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

        tenant = _normalise_optional(raw.get("tenant"), field="tenant")
        process = _normalise_optional(raw.get("process"), field="process")
        collection_id = _normalise_optional(
            raw.get("collection_id"), field="collection_id"
        )
        workflow_id = _normalise_optional(raw.get("workflow_id"), field="workflow_id")
        doc_class = _normalise_optional(raw.get("doc_class"), field="doc_class")
        chunker_mode = _normalise_optional(
            raw.get("chunker_mode"), field="chunker_mode"
        )

        if use_collection_routing and collection_id is None and doc_class is not None:
            collection_id = doc_class
            doc_class = None

        rule = RoutingRule(
            profile=profile,
            index=index,
            tenant=tenant,
            process=process,
            collection_id=collection_id,
            workflow_id=workflow_id,
            doc_class=doc_class,
            chunker_mode=chunker_mode,
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
    seen_selectors: dict[
        tuple[str | None, str | None, str | None, str | None, str | None], RoutingRule
    ] = {}
    duplicate_same_target: dict[
        tuple[str | None, str | None, str | None, str | None, str | None], RoutingRule
    ] = {}
    unique_rules: list[RoutingRule] = []

    for rule in rules:
        selector = rule.selector_tuple
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
            if left.priority_key != right.priority_key:
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
    if (
        left.tenant is not None
        and right.tenant is not None
        and left.tenant != right.tenant
    ):
        return False
    if (
        left.collection_id is not None
        and right.collection_id is not None
        and left.collection_id != right.collection_id
    ):
        return False
    if (
        left.workflow_id is not None
        and right.workflow_id is not None
        and left.workflow_id != right.workflow_id
    ):
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

    # Parse default_chunker_mode (optional, defaults to "late")
    default_chunker_mode = str(raw_config.get("default_chunker_mode", "late")).strip()
    if not default_chunker_mode:
        default_chunker_mode = "late"

    raw_rules = raw_config.get("rules", [])
    if not isinstance(raw_rules, Sequence):
        raise RoutingConfigurationError(
            _format_error(
                RoutingErrorCode.RULES_NOT_SEQUENCE,
                "rules must be a sequence of mappings",
            )
        )

    rules = _build_rules(raw_rules)

    table = RoutingTable(
        default_profile=default_profile,
        rules=rules,
        default_chunker_mode=default_chunker_mode,
    )

    config = get_embedding_configuration()

    _ensure_profiles_exist(
        table=table,
        available_profiles=config.embedding_profiles.keys(),
    )
    _validate_rule_uniqueness(table.rules)

    return table


def resolve_embedding_profile_id(
    *,
    tenant: str,
    process: str | None = None,
    doc_class: str | None = None,
    collection_id: str | None = None,
    workflow_id: str | None = None,
    language: str | None = None,
    size: str | None = None,
) -> str:
    from .profile_resolver import resolve_embedding_profile

    return resolve_embedding_profile(
        tenant_id=tenant,
        process=process,
        doc_class=doc_class,
        collection_id=collection_id,
        workflow_id=workflow_id,
        language=language,
        size=size,
    )


def resolve_chunker_mode(
    *,
    tenant: str,
    process: str | None = None,
    doc_class: str | None = None,
    collection_id: str | None = None,
    workflow_id: str | None = None,
) -> str:
    """
    Resolve chunker mode based on routing rules.

    Args:
        tenant: Tenant identifier
        process: Optional process name
        doc_class: Optional document class
        collection_id: Optional collection identifier
        workflow_id: Optional workflow identifier

    Returns:
        Chunker mode string (late | agentic | hybrid)

    Raises:
        RoutingConfigurationError: If routing configuration is invalid
    """
    table = get_routing_table()

    # Find best matching rule with chunker_mode
    best_rule: RoutingRule | None = None
    best_priority: tuple[int, int, int, int, int, int] | None = None

    for rule in table.rules:
        # Skip rules without chunker_mode
        if rule.chunker_mode is None:
            continue

        if not rule.matches(
            tenant=tenant,
            process=process,
            collection_id=collection_id,
            workflow_id=workflow_id,
            doc_class=doc_class,
        ):
            continue

        priority = rule.priority_key

        if best_priority is None or priority > best_priority:
            best_rule = rule
            best_priority = priority
            continue

        if priority == best_priority and best_rule is not None:
            if rule.chunker_mode != best_rule.chunker_mode:
                raise RoutingConfigurationError(
                    _format_error(
                        RoutingErrorCode.CONFLICT,
                        "Ambiguous routing rules match the same selector for chunker_mode",
                    )
                )

    # Return matched rule's chunker_mode or default
    if best_rule is not None:
        return best_rule.chunker_mode  # type: ignore[return-value]

    return table.default_chunker_mode


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
    "RoutingResolution",
    "RoutingErrorCode",
    "get_routing_table",
    "is_collection_routing_enabled",
    "resolve_embedding_profile_id",
    "resolve_chunker_mode",
    "reset_routing_rules_cache",
    "validate_routing_rules",
]
