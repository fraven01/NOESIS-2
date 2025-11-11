from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import yaml
from django.apps import apps
from django.conf import settings

logger = logging.getLogger(__name__)


class DomainPolicyAction(StrEnum):
    BOOST = "boost"
    REJECT = "reject"


@dataclass(frozen=True)
class DomainPolicyDecision:
    action: DomainPolicyAction
    priority: int
    source: str


@dataclass(frozen=True)
class DomainPolicyRule:
    action: DomainPolicyAction
    priority: int
    pattern: re.Pattern[str]
    source: str


@dataclass(frozen=True)
class _HostRule:
    domain: str
    action: DomainPolicyAction
    priority: int
    source: str


class DomainTrie:
    __slots__ = ("_root",)

    def __init__(self) -> None:
        self._root: dict[str, dict[str, object]] = {}

    def insert(self, domain: str, decision: DomainPolicyDecision) -> None:
        node = self._root
        for label in domain.split(".")[::-1]:
            node = node.setdefault(label, {})  # type: ignore[assignment]
        node.setdefault("_decisions", []).append(decision)  # type: ignore[assignment]

    def match(self, host: str) -> Sequence[DomainPolicyDecision]:
        node = self._root
        decisions: list[DomainPolicyDecision] = []
        for label in host.split(".")[::-1]:
            if label not in node:
                break
            node = node[label]  # type: ignore[index]
            if "_decisions" in node:
                decisions.extend(node["_decisions"])  # type: ignore[index]
        return decisions

    def copy(self) -> "DomainTrie":
        clone = DomainTrie()
        clone._root = copy.deepcopy(self._root)
        return clone


@dataclass
class DomainPolicy:
    trie: DomainTrie = field(default_factory=DomainTrie)
    regex_rules: list[DomainPolicyRule] = field(default_factory=list)
    boost_hosts: set[str] = field(default_factory=set)
    reject_hosts: set[str] = field(default_factory=set)

    def clone(self) -> "DomainPolicy":
        return DomainPolicy(
            trie=self.trie.copy(),
            regex_rules=list(self.regex_rules),
            boost_hosts=set(self.boost_hosts),
            reject_hosts=set(self.reject_hosts),
        )

    def add_host(
        self,
        host: str,
        action: DomainPolicyAction,
        *,
        priority: int,
        source: str,
    ) -> None:
        host_normalised = host.lower().strip()
        if not host_normalised:
            return
        decision = DomainPolicyDecision(action=action, priority=priority, source=source)
        self.trie.insert(host_normalised, decision)
        if action is DomainPolicyAction.BOOST:
            self.boost_hosts.add(host_normalised)
            self.reject_hosts.discard(host_normalised)
        else:
            self.reject_hosts.add(host_normalised)
            self.boost_hosts.discard(host_normalised)

    def evaluate(self, host: str | None) -> DomainPolicyDecision | None:
        if not host:
            return None
        host_normalised = host.lower().strip()
        if not host_normalised:
            return None
        decision: DomainPolicyDecision | None = None
        for candidate in self.trie.match(host_normalised):
            decision = _choose_decision(decision, candidate)
        for rule in self.regex_rules:
            if rule.pattern.search(host_normalised):
                decision = _choose_decision(
                    decision,
                    DomainPolicyDecision(
                        action=rule.action,
                        priority=rule.priority,
                        source=rule.source,
                    ),
                )
        return decision

    def is_blocked(self, host: str | None) -> bool:
        decision = self.evaluate(host)
        return bool(decision and decision.action is DomainPolicyAction.REJECT)


def _choose_decision(
    current: DomainPolicyDecision | None, candidate: DomainPolicyDecision
) -> DomainPolicyDecision:
    if current is None:
        return candidate
    if candidate.priority > current.priority:
        return candidate
    if candidate.priority < current.priority:
        return current
    if candidate.action is DomainPolicyAction.REJECT and current.action is DomainPolicyAction.BOOST:
        return candidate
    return current


def _normalise_host(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = urlsplit(text)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
    else:
        host = text
    if not host:
        return None
    return host.lower().strip()


def _priority_default(action: DomainPolicyAction) -> int:
    return 90 if action is DomainPolicyAction.REJECT else 70


def _convert_wildcard(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{escaped}$", re.IGNORECASE)


def _extract_host_rules(
    values: Iterable[object] | None,
    *,
    action: DomainPolicyAction,
    source: str,
) -> tuple[list[_HostRule], list[DomainPolicyRule]]:
    host_rules: list[_HostRule] = []
    regex_rules: list[DomainPolicyRule] = []
    for item in values or []:
        if isinstance(item, Mapping):
            host_value = item.get("host") or item.get("domain") or item.get("value")
            priority_raw = item.get("priority")
        else:
            host_value = item
            priority_raw = None
        host = _normalise_host(host_value)
        if not host:
            continue
        priority = _priority_default(action)
        if isinstance(priority_raw, (int, float)):
            priority = int(priority_raw)
        elif isinstance(priority_raw, str) and priority_raw.strip():
            try:
                priority = int(priority_raw.strip())
            except ValueError:
                logger.debug("hybrid.policy_priority_invalid", extra={"value": priority_raw})
        if "*" in host or "?" in host:
            pattern = _convert_wildcard(host)
            regex_rules.append(
                DomainPolicyRule(action=action, priority=priority, pattern=pattern, source=source)
            )
        else:
            host_rules.append(
                _HostRule(domain=host, action=action, priority=priority, source=source)
            )
    return host_rules, regex_rules


def _extract_regex_rules(
    values: Iterable[Mapping[str, object]] | None,
    *,
    source: str,
) -> list[DomainPolicyRule]:
    rules: list[DomainPolicyRule] = []
    for item in values or []:
        if not isinstance(item, Mapping):
            continue
        pattern_raw = item.get("pattern") or item.get("regex")
        if not isinstance(pattern_raw, str):
            continue
        action_raw = str(item.get("action") or DomainPolicyAction.BOOST.value)
        try:
            action = DomainPolicyAction(action_raw)
        except ValueError:
            logger.debug("hybrid.policy_action_invalid", extra={"action": action_raw})
            action = DomainPolicyAction.BOOST
        priority = _priority_default(action)
        priority_raw = item.get("priority")
        if isinstance(priority_raw, (int, float)):
            priority = int(priority_raw)
        elif isinstance(priority_raw, str) and priority_raw.strip():
            try:
                priority = int(priority_raw.strip())
            except ValueError:
                logger.debug("hybrid.policy_priority_invalid", extra={"value": priority_raw})
        try:
            pattern = re.compile(pattern_raw, re.IGNORECASE)
        except re.error as exc:
            logger.debug("hybrid.policy_regex_invalid", extra={"error": str(exc)})
            continue
        rules.append(
            DomainPolicyRule(action=action, priority=priority, pattern=pattern, source=source)
        )
    return rules


@dataclass(frozen=True)
class _YamlPolicy:
    hosts: list[_HostRule]
    regex: list[DomainPolicyRule]


@dataclass(frozen=True)
class _YamlBundle:
    default: _YamlPolicy
    tenants: Mapping[str, _YamlPolicy]


def _load_yaml_defaults() -> _YamlBundle:
    path_setting = getattr(settings, "HYBRID_DOMAIN_POLICY_PATH", None)
    path = Path(path_setting) if path_setting else Path(settings.BASE_DIR) / "config" / "domain_policies.yaml"
    if not path.exists():
        return _YamlBundle(_YamlPolicy([], []), {})
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("hybrid.policy_yaml_load_failed", extra={"error": str(exc)})
        return _YamlBundle(_YamlPolicy([], []), {})

    defaults_payload = data.get("defaults") or {}
    tenants_payload = data.get("tenants") or {}

    default_hosts_preferred, default_regex_preferred = _extract_host_rules(
        defaults_payload.get("preferred") or defaults_payload.get("boost"),
        action=DomainPolicyAction.BOOST,
        source="yaml.defaults.preferred",
    )
    default_hosts_blocked, default_regex_blocked = _extract_host_rules(
        defaults_payload.get("blocked") or defaults_payload.get("reject"),
        action=DomainPolicyAction.REJECT,
        source="yaml.defaults.blocked",
    )
    default_regex_rules = list(default_regex_preferred) + list(default_regex_blocked)
    default_regex_rules.extend(
        _extract_regex_rules(defaults_payload.get("rules"), source="yaml.defaults.rules")
    )
    default_hosts = default_hosts_preferred + default_hosts_blocked

    tenant_map: dict[str, _YamlPolicy] = {}
    for tenant_id, payload in tenants_payload.items():
        if not tenant_id:
            continue
        hosts_boost, regex_boost = _extract_host_rules(
            (payload or {}).get("preferred") or (payload or {}).get("boost"),
            action=DomainPolicyAction.BOOST,
            source=f"yaml.tenant.{tenant_id}.preferred",
        )
        hosts_block, regex_block = _extract_host_rules(
            (payload or {}).get("blocked") or (payload or {}).get("reject"),
            action=DomainPolicyAction.REJECT,
            source=f"yaml.tenant.{tenant_id}.blocked",
        )
        regex_rules = list(regex_boost) + list(regex_block)
        regex_rules.extend(
            _extract_regex_rules(
                (payload or {}).get("rules"), source=f"yaml.tenant.{tenant_id}.rules"
            )
        )
        tenant_map[str(tenant_id)] = _YamlPolicy(hosts=hosts_boost + hosts_block, regex=regex_rules)

    return _YamlBundle(_YamlPolicy(default_hosts, default_regex_rules), tenant_map)


def _load_override_lists(tenant_id: str | None) -> list[_HostRule]:
    if not tenant_id:
        return []
    if not apps.is_installed("common"):
        return []
    try:
        model = apps.get_model("common", "DomainPolicyOverride")
    except LookupError:  # pragma: no cover - defensive guard
        return []
    try:
        record = model.objects.filter(tenant_id=tenant_id).values("preferred_hosts", "blocked_hosts").first()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("hybrid.policy_override_query_failed", extra={"error": str(exc)})
        return []
    if not record:
        return []
    hosts: list[_HostRule] = []
    preferred_hosts, _ = _extract_host_rules(
        record.get("preferred_hosts"),
        action=DomainPolicyAction.BOOST,
        source="db.override.preferred",
    )
    blocked_hosts, _ = _extract_host_rules(
        record.get("blocked_hosts"),
        action=DomainPolicyAction.REJECT,
        source="db.override.blocked",
    )
    hosts.extend(preferred_hosts)
    hosts.extend(blocked_hosts)
    return hosts


def _load_db_rules(tenant_id: str | None) -> tuple[list[_HostRule], list[DomainPolicyRule]]:
    if not tenant_id:
        return [], []
    if not apps.is_installed("common"):
        return [], []
    try:
        model = apps.get_model("common", "DomainPolicy")
    except LookupError:  # pragma: no cover - defensive guard
        return [], []
    try:
        rows = model.objects.filter(tenant_id=tenant_id).values("domain", "action", "priority")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("hybrid.policy_rule_query_failed", extra={"error": str(exc)})
        return [], []
    host_rules: list[_HostRule] = []
    regex_rules: list[DomainPolicyRule] = []
    for row in rows:
        domain = _normalise_host(row.get("domain"))
        if not domain:
            continue
        action_raw = row.get("action") or DomainPolicyAction.BOOST.value
        try:
            action = DomainPolicyAction(str(action_raw))
        except ValueError:
            logger.debug("hybrid.policy_action_invalid", extra={"action": action_raw})
            action = DomainPolicyAction.BOOST
        priority = row.get("priority")
        priority_value = _priority_default(action)
        if isinstance(priority, (int, float)):
            priority_value = int(priority)
        if "*" in domain or "?" in domain:
            regex_rules.append(
                DomainPolicyRule(
                    action=action,
                    priority=priority_value,
                    pattern=_convert_wildcard(domain),
                    source="db.rule",
                )
            )
        else:
            host_rules.append(
                _HostRule(
                    domain=domain,
                    action=action,
                    priority=priority_value,
                    source="db.rule",
                )
            )
    return host_rules, regex_rules


def _build_policy(host_rules: Sequence[_HostRule], regex_rules: Sequence[DomainPolicyRule]) -> DomainPolicy:
    policy = DomainPolicy()
    for rule in host_rules:
        policy.add_host(
            rule.domain,
            rule.action,
            priority=rule.priority,
            source=rule.source,
        )
    policy.regex_rules.extend(regex_rules)
    return policy


_CACHE_TTL_S = 3600
_POLICY_CACHE: dict[str, tuple[float, DomainPolicy]] = {}


def get_domain_policy(tenant_id: str | None) -> DomainPolicy:
    cache_key = tenant_id or "__default__"
    now = monotonic()
    cached = _POLICY_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_S:
        return cached[1]

    yaml_bundle = _load_yaml_defaults()
    host_rules = list(yaml_bundle.default.hosts)
    regex_rules = list(yaml_bundle.default.regex)
    if tenant_id and tenant_id in yaml_bundle.tenants:
        tenant_policy = yaml_bundle.tenants[tenant_id]
        host_rules.extend(tenant_policy.hosts)
        regex_rules.extend(tenant_policy.regex)

    host_rules.extend(_load_override_lists(tenant_id))
    db_host_rules, db_regex_rules = _load_db_rules(tenant_id)
    host_rules.extend(db_host_rules)
    regex_rules.extend(db_regex_rules)

    policy = _build_policy(host_rules, regex_rules)
    _POLICY_CACHE[cache_key] = (now, policy)
    return policy


__all__ = [
    "DomainPolicy",
    "DomainPolicyAction",
    "DomainPolicyDecision",
    "get_domain_policy",
]

