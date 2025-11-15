"""Context helpers for document and asset contract validation toggles."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Optional, Tuple

from .contract_utils import normalize_string


_STRICT_CHECKSUMS: ContextVar[bool] = ContextVar("STRICT_CHECKSUMS", default=False)
_ASSET_MEDIA_GUARD: ContextVar[Optional[Tuple[str, ...]]] = ContextVar(
    "ASSET_MEDIA_GUARD", default=None
)


def set_strict_checksums(enabled: bool = True) -> None:
    """Toggle strict checksum verification for contract models."""

    _STRICT_CHECKSUMS.set(bool(enabled))


@contextmanager
def strict_checksums(enabled: bool = True):
    """Context manager enabling strict checksum validation within the scope."""

    token = _STRICT_CHECKSUMS.set(bool(enabled))
    try:
        yield
    finally:
        _STRICT_CHECKSUMS.reset(token)


def is_strict_checksums_enabled() -> bool:
    """Return whether strict checksum validation is enabled for the current context."""

    return bool(_STRICT_CHECKSUMS.get())


def _normalize_guard_prefix(value: str) -> str:
    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("media_guard_empty")
    return normalized.lower()


def set_asset_media_guard(prefixes: Optional[Iterable[str]]) -> None:
    """Configure allowed media type prefixes for non-inline asset blobs."""

    normalized: Optional[Tuple[str, ...]] = None
    if prefixes:
        normalized = tuple(_normalize_guard_prefix(prefix) for prefix in prefixes)
    _ASSET_MEDIA_GUARD.set(normalized)


@contextmanager
def asset_media_guard(prefixes: Optional[Iterable[str]]):
    """Context manager enforcing non-inline asset media type prefixes."""

    normalized: Optional[Tuple[str, ...]]
    if prefixes:
        normalized = tuple(_normalize_guard_prefix(prefix) for prefix in prefixes)
    else:
        normalized = None
    token = _ASSET_MEDIA_GUARD.set(normalized)
    try:
        yield
    finally:
        _ASSET_MEDIA_GUARD.reset(token)


def get_asset_media_guard() -> Optional[Tuple[str, ...]]:
    """Return the configured asset media guard prefixes for the current context."""

    return _ASSET_MEDIA_GUARD.get()
