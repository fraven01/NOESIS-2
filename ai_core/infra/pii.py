from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.policy import (
    derive_scoped_hmac_key,
    detect_names,
    get_policy_rules,
    get_session_scope,
)
from ai_core.metrics.pii_metrics import record_detection


_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(
            r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.+-])",
            re.IGNORECASE,
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(\+?\d[\d\s().-]{5,}\d)(?!\w)",
        ),
    ),
    (
        "IBAN",
        re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b", re.IGNORECASE),
    ),
    (
        "IPV4",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "IPV6",
        re.compile(r"\b(?:[0-9A-F]{1,4}:){2,7}[0-9A-F]{1,4}\b", re.IGNORECASE),
    ),
    (
        "UUID",
        re.compile(
            r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
        ),
    ),
    (
        "NUMBER",
        re.compile(r"\b\d{3,}\b"),
    ),
)

_SENSITIVE_JSON_KEYS = {
    "type",
    "client_email",
    "private_key_id",
    "private_key",
    "client_id",
    "access_token",
    "refresh_token",
}

_SENSITIVE_QUERY_PARAMS = {
    "email",
    "phone",
    "token",
    "code",
    "apikey",
    "api_key",
    "api-key",
    "access_key",
    "access-key",
    "access_token",
    "access-token",
    "id_token",
    "client_secret",
    "secret",
    "password",
    "pass",
    "pwd",
    "session",
    "session_id",
    "sessionid",
    "sid",
    "auth",
    "authorization",
}

_HIGH_ENTROPY_URLSAFE = re.compile(r"^[A-Za-z0-9_-]{20,}$")
_HIGH_ENTROPY_BASE64 = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")
_HIGH_ENTROPY_HEX = re.compile(r"^[A-Fa-f0-9]{32,}$")

_JWT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_JWT_INLINE_PATTERN = re.compile(
    r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)

_URL_PATTERN = re.compile(r"https?://[^\s)]+")


def _looks_masked(value: str) -> bool:
    return bool(
        re.match(r"^\[REDACTED(?:_[A-Z0-9_]+)?\]$", value)
        or re.match(r"^<[A-Z0-9_]+_[0-9a-fA-F]{8}>$", value)
    )


def _is_high_entropy_secret(value: str) -> bool:
    return bool(
        _HIGH_ENTROPY_URLSAFE.match(value)
        or _HIGH_ENTROPY_BASE64.match(value)
        or _HIGH_ENTROPY_HEX.match(value)
    )


def _make_placeholder(
    tag: str,
    original: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    session_scope: Optional[Iterable[str]] = None,
) -> str:
    tag = tag.upper()
    if deterministic and hmac_key:
        scoped_key = derive_scoped_hmac_key(hmac_key, session_scope)
        digest = hmac.new(scoped_key, original.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
        return f"<{tag}_{digest}>"
    return f"[REDACTED_{tag}]"


def mask_text(
    text: str,
    policy: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    mode: str = "industrial",
    name_detection: bool = False,
    session_scope: Optional[Iterable[str]] = None,
    structured_max_length: Optional[int] = None,
    json_dump_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Mask PII in free-form text using regex detectors."""

    if not text or not isinstance(text, str):
        return text

    if str(policy).lower() == "off" or str(mode).lower() == "off":
        return text

    rules = get_policy_rules(policy, mode, name_detection=name_detection)
    scoped_session = session_scope or get_session_scope()

    masked_text = mask_structured(
        text,
        policy,
        deterministic,
        hmac_key,
        mode=mode,
        name_detection=name_detection,
        session_scope=scoped_session,
        structured_max_length=structured_max_length,
        json_dump_kwargs=json_dump_kwargs,
    )

    if rules is None or "JWT" in rules:
        masked_text = _JWT_INLINE_PATTERN.sub(
            lambda match: _mask_jwt(
                match.group(0),
                deterministic,
                hmac_key,
                mode=mode,
                session_scope=scoped_session,
            ),
            masked_text,
        )

    masked_text = _mask_urls_in_text(
        masked_text,
        policy,
        deterministic,
        hmac_key,
        rules=rules,
        mode=mode,
        session_scope=scoped_session,
    )

    for tag, pattern in _PII_PATTERNS:
        if rules is None and tag == "NUMBER":
            continue
        if rules is not None and tag not in rules:
            continue

        def _replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if _looks_masked(value):
                return value
            placeholder = _make_placeholder(
                tag,
                value,
                deterministic,
                hmac_key,
                session_scope=scoped_session,
            )
            if mode == "gold":
                record_detection(tag)
            return placeholder

        masked_text = pattern.sub(_replace, masked_text)

    if mode == "gold" and rules is not None and "NAME" in rules and name_detection:
        for candidate in detect_names(masked_text):
            if _looks_masked(candidate):
                continue
            placeholder = _make_placeholder(
                "NAME",
                candidate,
                deterministic,
                hmac_key,
                session_scope=scoped_session,
            )
            masked_text = masked_text.replace(candidate, placeholder)
            record_detection("NAME")

    return masked_text


def mask_json(
    obj: Any,
    policy: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    mode: str = "industrial",
    name_detection: bool = False,
    session_scope: Optional[Iterable[str]] = None,
) -> Any:
    """Recursively mask sensitive fields in JSON-like structures."""

    rules = get_policy_rules(policy, mode, name_detection=name_detection)
    scoped_session = session_scope or get_session_scope()

    if isinstance(obj, dict):
        result: dict[Any, Any] = {}
        for key, value in obj.items():
            tag = key.upper()
            is_sensitive_key = key in _SENSITIVE_JSON_KEYS
            if key == "type":
                sa_like = (
                    "private_key" in obj
                    or "client_email" in obj
                    or (isinstance(value, str) and value == "service_account")
                )
                if not sa_like:
                    is_sensitive_key = False
            if rules is not None and tag not in rules and is_sensitive_key:
                # Policy disabled masking for this class.
                result[key] = value
                continue
            if is_sensitive_key and isinstance(value, str) and _looks_masked(value):
                result[key] = value
            elif is_sensitive_key:
                result[key] = _make_placeholder(
                    key,
                    str(value),
                    deterministic,
                    hmac_key,
                    session_scope=scoped_session,
                )
                if mode == "gold":
                    record_detection(tag)
            else:
                result[key] = mask_json(
                    value,
                    policy,
                    deterministic,
                    hmac_key,
                    mode=mode,
                    name_detection=name_detection,
                    session_scope=scoped_session,
                )
        return result
    if isinstance(obj, list):
        return [
            mask_json(
                item,
                policy,
                deterministic,
                hmac_key,
                mode=mode,
                name_detection=name_detection,
                session_scope=scoped_session,
            )
            for item in obj
        ]
    if isinstance(obj, str):
        return mask_text(
            obj,
            policy,
            deterministic,
            hmac_key,
            mode=mode,
            name_detection=name_detection,
            session_scope=scoped_session,
        )
    return obj


def _mask_jwt(
    token: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    mode: str = "industrial",
    session_scope: Optional[Iterable[str]] = None,
) -> str:
    if token.count(".") != 2:
        return token
    header, payload, signature = token.split(".")
    if all(_looks_masked(segment) for segment in (header, payload, signature)):
        return token
    placeholder = ".".join(
        (
            _make_placeholder(
                "JWT_HEADER",
                header,
                deterministic,
                hmac_key,
                session_scope=session_scope,
            ),
            _make_placeholder(
                "JWT_PAYLOAD",
                payload,
                deterministic,
                hmac_key,
                session_scope=session_scope,
            ),
            _make_placeholder(
                "JWT_SIG",
                signature,
                deterministic,
                hmac_key,
                session_scope=session_scope,
            ),
        )
    )
    if mode == "gold":
        record_detection("JWT")
    return placeholder


def _mask_url(
    text: str,
    policy: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    mode: str = "industrial",
    session_scope: Optional[Iterable[str]] = None,
    rules: Optional[Mapping[str, Any]] = None,
) -> str:
    parsed = urlparse(text)
    if not parsed.query:
        return text
    params = parse_qsl(parsed.query, keep_blank_values=True)
    masked_params = []
    effective_rules = rules if rules is not None else get_policy_rules(policy, mode)
    for key, value in params:
        lower_key = key.lower()
        tag = key.upper()

        def _should_mask() -> bool:
            if _looks_masked(value):
                return False
            if effective_rules is not None and tag not in effective_rules:
                return False
            if lower_key in _SENSITIVE_QUERY_PARAMS:
                return True
            if lower_key == "key":
                return _is_high_entropy_secret(value)
            return False

        if _should_mask():
            masked_params.append(
                (
                    key,
                    _make_placeholder(
                        key,
                        value,
                        deterministic,
                        hmac_key,
                        session_scope=session_scope,
                    ),
                )
            )
            if mode == "gold":
                record_detection(tag)
        else:
            masked_params.append((key, value))
    masked_query = urlencode(masked_params, doseq=True, safe="/@")
    return urlunparse(parsed._replace(query=masked_query))


def _mask_urls_in_text(
    text: str,
    policy: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    rules: Optional[Mapping[str, Any]] = None,
    mode: str = "industrial",
    session_scope: Optional[Iterable[str]] = None,
) -> str:
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        return _mask_url(
            match.group(0),
            policy,
            deterministic,
            hmac_key,
            mode=mode,
            session_scope=session_scope,
            rules=rules,
        )

    return _URL_PATTERN.sub(_replace, text)


def mask_structured(
    text: str,
    policy: str,
    deterministic: bool,
    hmac_key: Optional[bytes],
    *,
    mode: str = "industrial",
    name_detection: bool = False,
    session_scope: Optional[Iterable[str]] = None,
    structured_max_length: Optional[int] = None,
    json_dump_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Detect structured inputs (JWT, JSON, URLs) and mask accordingly."""

    rules = get_policy_rules(policy, mode, name_detection=name_detection)

    if not text:
        return text

    scoped_session = session_scope or get_session_scope()

    stripped = text.strip()
    if _JWT_PATTERN.match(stripped):
        if rules is not None and "JWT" not in rules:
            return text
        return _mask_jwt(
            stripped,
            deterministic,
            hmac_key,
            mode=mode,
            session_scope=scoped_session,
        )

    should_skip_json = bool(
        structured_max_length is not None and len(text) > structured_max_length
    )

    if not should_skip_json:
        try:
            loaded = json.loads(text)
        except (TypeError, ValueError):
            pass
        else:
            masked = mask_json(
                loaded,
                policy,
                deterministic,
                hmac_key,
                mode=mode,
                name_detection=name_detection,
                session_scope=scoped_session,
            )
            dumps_kwargs: dict[str, Any] = {"separators": (",", ":")}
            if json_dump_kwargs:
                dumps_kwargs.update(json_dump_kwargs)
            if dumps_kwargs.get("separators") is None:
                dumps_kwargs.pop("separators")
            return json.dumps(masked, **dumps_kwargs)

    parsed = urlparse(text)
    if parsed.scheme or parsed.query:
        return _mask_url(
            text,
            policy,
            deterministic,
            hmac_key,
            mode=mode,
            session_scope=scoped_session,
            rules=rules,
        )

    return text


def mask(text: str) -> str:
    """Legacy masking that defaults to balanced policy without determinism."""

    config = get_pii_config()
    mode = str(config.get("mode", "")).lower()
    policy = str(config.get("policy", "balanced"))
    if mode == "off" or policy.lower() == "off":
        return text
    deterministic = bool(config.get("deterministic"))
    hmac_key = config.get("hmac_secret") if deterministic else None
    return mask_text(
        text,
        policy,
        deterministic,
        hmac_key if isinstance(hmac_key, (bytes, bytearray)) else None,
        mode=config.get("mode", "industrial"),
        name_detection=bool(config.get("name_detection", False)),
        session_scope=config.get("session_scope"),
    )


def mask_prompt(text: str, *, placeholder_only: bool = False) -> str:
    """Backward-compatible entrypoint to the prompt masking utilities."""

    from ai_core.infra.mask_prompt import mask_prompt as _mask_prompt

    return _mask_prompt(text, placeholder_only=placeholder_only)
