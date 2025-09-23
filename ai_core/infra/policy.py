"""Policy helpers and session scoping for the PII masking pipeline."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class PolicyRule:
    """Rule definition for a PII class within a policy profile."""

    tag: str
    format_preserving: bool = False


_SESSION_SCOPE: ContextVar[Optional[tuple[str, ...]]] = ContextVar(
    "pii_session_scope", default=None
)


_FORMAT_PRESERVING_TAGS = {
    "PHONE",
}


_GOLD_POLICY_MATRIX: Mapping[str, frozenset[str]] = {
    "strict": frozenset(
        {
            "EMAIL",
            "PHONE",
            "IBAN",
            "IPV4",
            "IPV6",
            "UUID",
            "JWT",
            "NUMBER",
            # Structured secrets
            "TYPE",
            "CLIENT_EMAIL",
            "PRIVATE_KEY_ID",
            "PRIVATE_KEY",
            "CLIENT_ID",
            "ACCESS_TOKEN",
            "ACCESS-TOKEN",
            "REFRESH_TOKEN",
            # Query params
            "EMAIL",
            "PHONE",
            "TOKEN",
            "CODE",
            "KEY",
            "APIKEY",
            "API_KEY",
            "API-KEY",
            "ACCESS_KEY",
            "ACCESS-KEY",
            "PASSWORD",
            "PASS",
            "PWD",
            "SECRET",
            "SESSION",
            "SESSION_ID",
            "SESSIONID",
            "SID",
            "ID_TOKEN",
            "CLIENT_SECRET",
            "AUTH",
            "AUTHORIZATION",
        }
    ),
    "balanced": frozenset(
        {
            "EMAIL",
            "PHONE",
            "UUID",
            "JWT",
            "TYPE",
            "CLIENT_EMAIL",
            "PRIVATE_KEY",
            "ACCESS_TOKEN",
            "ACCESS-TOKEN",
            "REFRESH_TOKEN",
            "TOKEN",
            "CODE",
            "PASSWORD",
            "SECRET",
            "APIKEY",
            "API_KEY",
            "API-KEY",
            "ACCESS_KEY",
            "ACCESS-KEY",
            "ID_TOKEN",
            "CLIENT_SECRET",
            "SESSION",
            "SESSION_ID",
            "SESSIONID",
            "SID",
            "PWD",
            "AUTH",
            "AUTHORIZATION",
        }
    ),
    "lenient": frozenset(
        {
            "EMAIL",
            "PHONE",
        }
    ),
}


def _ensure_tuple(scope: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in scope if value is not None)


def set_session_scope(*, tenant_id: str, case_id: str, session_salt: str) -> None:
    """Persist the active masking scope for session-stable tokens."""

    _SESSION_SCOPE.set(
        _ensure_tuple((tenant_id, case_id, session_salt)) or None
    )


def clear_session_scope() -> None:
    """Reset the active session scope."""

    _SESSION_SCOPE.set(None)


def get_session_scope() -> Optional[tuple[str, ...]]:
    """Return the currently active session scope if available."""

    return _SESSION_SCOPE.get()


def derive_scoped_hmac_key(
    hmac_key: bytes, session_scope: Optional[Iterable[str]]
) -> bytes:
    """Derive a deterministic HMAC key that stays stable within a session."""

    if not session_scope:
        return hmac_key
    scope_tuple = _ensure_tuple(session_scope)
    if not scope_tuple:
        return hmac_key
    import hmac
    import hashlib

    scope_bytes = "|".join(scope_tuple).encode("utf-8")
    return hmac.new(hmac_key, scope_bytes, hashlib.sha256).digest()


def get_policy_rules(
    policy: str,
    mode: str,
    *,
    name_detection: bool = False,
) -> Optional[Mapping[str, PolicyRule]]:
    """Return active policy rules for gold mode or ``None`` for industrial mode."""

    if mode.lower() != "gold":
        return None

    base = _GOLD_POLICY_MATRIX.get(policy, _GOLD_POLICY_MATRIX["balanced"])
    rules: MutableMapping[str, PolicyRule] = {}
    for tag in base:
        rules[tag] = PolicyRule(tag=tag, format_preserving=tag in _FORMAT_PRESERVING_TAGS)
    if name_detection:
        rules["NAME"] = PolicyRule(tag="NAME")
    return dict(rules)


def detect_names(text: str) -> list[str]:
    """Placeholder hook for future name detection (NER integration)."""

    return []


__all__ = [
    "PolicyRule",
    "clear_session_scope",
    "derive_scoped_hmac_key",
    "detect_names",
    "get_policy_rules",
    "get_session_scope",
    "set_session_scope",
]
