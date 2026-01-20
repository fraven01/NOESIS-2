"""Shared helpers for chunking utilities."""

from __future__ import annotations

import re
from typing import Iterable, List


_DOT_PLACEHOLDER = "__DOT__"

_ABBREVIATIONS = (
    "dr",
    "mr",
    "ms",
    "mrs",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
)

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)
_INITIALS_PATTERN = re.compile(r"\b(?:[A-Z]\.){2,}")
_DECIMAL_PATTERN = re.compile(r"\b\d+\.\d+\b")
_URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s\)\]\}\.,!?]+(?:\.[^\s\)\]\}\.,!?]+)*",
    re.IGNORECASE,
)


def _protect_dots(text: str, patterns: Iterable[re.Pattern[str]]) -> str:
    for pattern in patterns:
        text = pattern.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), text)
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with lightweight protections."""
    if not text:
        return []

    protected = _protect_dots(
        text,
        (
            _URL_PATTERN,
            _DECIMAL_PATTERN,
            _INITIALS_PATTERN,
            _ABBREV_PATTERN,
        ),
    )
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    sentences = []
    for part in parts:
        restored = part.replace(_DOT_PLACEHOLDER, ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


__all__ = ["split_sentences"]
