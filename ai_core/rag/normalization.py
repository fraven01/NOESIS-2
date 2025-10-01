"""Utilities for consistent text normalisation used by RAG pipelines."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

__all__ = ["normalise_text", "normalise_text_db"]

_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_WORD_RE = re.compile(r"[\wäöüÄÖÜß]+", re.UNICODE)


def _strip_german_plural(token: str) -> str:
    """Apply a lightweight plural heuristic for German nouns."""

    candidate = token
    if len(candidate) <= 4:
        return candidate
    if not _WORD_RE.fullmatch(candidate):
        return candidate
    for suffix in ("en", "n", "e"):
        if candidate.endswith(suffix) and len(candidate) - len(suffix) >= 4:
            return candidate[: -len(suffix)]
    return candidate


def _apply_plural_heuristic(tokens: Iterable[str]) -> list[str]:
    return [_strip_german_plural(token) for token in tokens]


def normalise_text_db(value: str | None) -> str:
    """Replicate the text normalisation performed in PostgreSQL."""

    if not value:
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text)


def normalise_text(value: str | None) -> str:
    """Return a normalised representation of ``value`` for embeddings/search."""

    if not value:
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    tokens = text.split(" ")
    return " ".join(_apply_plural_heuristic(tokens))
