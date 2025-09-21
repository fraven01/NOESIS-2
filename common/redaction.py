"""Structured log redaction utilities."""

from __future__ import annotations

import hmac
import re
from hashlib import sha256
from typing import Mapping, MutableMapping

import structlog

MASK = "[REDACTED]"


def _get_hash_salt() -> bytes:
    from django.conf import settings  # imported lazily

    salt = getattr(settings, "LOG_HASH_SALT", "")
    if salt is None:
        salt = ""
    if isinstance(salt, bytes):
        return salt
    return str(salt).encode("utf-8")


def hash_str(value: str | bytes) -> str:
    """Return a salted HMAC-SHA256 digest for the given value."""

    if isinstance(value, bytes):
        payload = value
    else:
        payload = str(value).encode("utf-8")
    return hmac.new(_get_hash_salt(), payload, sha256).hexdigest()


def hash_email(email: str) -> str:
    """Normalise and hash an email address."""

    normalised = email.strip().lower()
    return hash_str(normalised)


def hash_user_id(user_id: str | int) -> str:
    """Hash a user identifier to preserve correlation without PII exposure."""

    return hash_str(str(user_id))


class Redactor:
    """Callable structlog processor that masks sensitive data."""

    _SENSITIVE_KEYWORDS: tuple[str, ...] = (
        "email",
        "phone",
        "bearer",
        "api_key",
        "apikey",
        "password",
        "token",
        "secret",
        "authorization",
        "auth_header",
        "private_key",
        "client_secret",
    )

    _PROMPT_FIELDS: tuple[str, ...] = ("prompt", "response")

    _SUB_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE), MASK),
        (re.compile(r"\b\+?[0-9]{1,3}[0-9()\s.-]{5,}[0-9]\b"), MASK),
        (re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b"), MASK),
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), MASK),
        (
            re.compile(
                r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
            ),
            MASK,
        ),
        (
            re.compile(r"-----BEGIN PRIVATE KEY-----[\s\S]+?-----END PRIVATE KEY-----"),
            MASK,
        ),
        (
            re.compile(r'("type"\s*:\s*)"service_account"', re.IGNORECASE),
            r"\1\"[REDACTED]\"",
        ),
        (
            re.compile(r'("client_secret"\s*:\s*)"[^"]+"', re.IGNORECASE),
            r"\1\"[REDACTED]\"",
        ),
        (re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{10,}\b"), MASK),
    )

    _CARD_PATTERN = re.compile(r"(?:\d[ -]?){13,19}")

    def __init__(self, mask: str = MASK) -> None:
        self.mask = mask

    def __call__(
        self,
        _: structlog.typing.WrappedLogger,
        __: str,
        event_dict: MutableMapping[str, object],
    ) -> MutableMapping[str, object]:
        return self._redact_mapping(event_dict)

    def _redact_mapping(self, mapping: MutableMapping[str, object]) -> MutableMapping[str, object]:
        for key, value in list(mapping.items()):
            mapping[key] = self._redact_value(key, value)
        return mapping

    def _redact_value(self, key: str | None, value: object) -> object:
        if isinstance(value, str):
            return self._redact_string(key, value)
        if isinstance(value, Mapping):
            mutable = dict(value)
            redacted = self._redact_mapping(mutable)
            if type(value) is dict:
                return redacted
            try:
                return type(value)(redacted)
            except Exception:
                return redacted
        if isinstance(value, list):
            return [self._redact_value(key, item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(key, item) for item in value)
        if isinstance(value, set):
            return {self._redact_value(key, item) for item in value}
        return value

    def _key_is_sensitive(self, key: str) -> bool:
        lowered = key.lower()
        return any(token in lowered for token in self._SENSITIVE_KEYWORDS)

    def _redact_string(self, key: str | None, value: str) -> str:
        if key and key.lower() in self._PROMPT_FIELDS and not self._log_llm_text_enabled():
            return self._redact_llm_field(value)

        if key and self._key_is_sensitive(key):
            return self.mask

        if self._requires_full_mask(value):
            return self.mask

        redacted = value
        for pattern, replacement in self._SUB_PATTERNS:
            redacted = pattern.sub(replacement, redacted)

        redacted = self._mask_cards(redacted)

        return redacted

    def _requires_full_mask(self, value: str) -> bool:
        return any(
            pattern.search(value)
            for pattern in (
                re.compile(r"^\s*-----BEGIN PRIVATE KEY-----", re.IGNORECASE),
                re.compile(r"^\s*eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\s*$"),
                re.compile(r"^\s*Bearer\s+[A-Za-z0-9\-._~+/]+=*\s*$", re.IGNORECASE),
                re.compile(r"^\s*[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\s*$"),
            )
        )

    def _redact_llm_field(self, value: str) -> str:
        digest = hash_str(value)
        length = len(value)
        return f"{self.mask} len={length} hash={digest}"

    @staticmethod
    def _log_llm_text_enabled() -> bool:
        from django.conf import settings  # imported lazily

        if getattr(settings, "configured", False):
            return getattr(settings, "LOG_LLM_TEXT", True)
        return True

    def _mask_cards(self, value: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            digits = re.sub(r"[^0-9]", "", match.group())
            if len(digits) < 12:
                return match.group()
            if self._luhn_valid(digits):
                return self.mask
            return match.group()

        return self._CARD_PATTERN.sub(_replace, value)

    @staticmethod
    def _luhn_valid(number: str) -> bool:
        total = 0
        reverse_digits = number[::-1]
        for idx, char in enumerate(reverse_digits):
            digit = ord(char) - 48
            if idx % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

