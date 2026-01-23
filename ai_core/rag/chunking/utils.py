"""Shared helpers for chunking utilities."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


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
_NUMBERED_LIST_ITEM_RE = re.compile(r"^\s*\(?(?P<num>\d{1,4})\)?[.)]\s+")


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
    protected = re.sub(
        r"(?:(?<=[:\n\r])|(?<=\s\s))\s*(\(?\d{1,4}\)?[.)]\s+)",
        r"\n\1",
        protected,
    )
    parts = re.split(r"(?<=[.!?])\s+|\n+", protected.strip())
    sentences = []
    for part in parts:
        restored = part.replace(_DOT_PLACEHOLDER, ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


def extract_numbered_list_index(text: str) -> int | None:
    """Return the leading list number if the text starts with a numbered item."""
    if not text:
        return None
    match = _NUMBERED_LIST_ITEM_RE.match(text)
    if not match:
        return None
    try:
        return int(match.group("num"))
    except (TypeError, ValueError):
        return None


def is_numbered_list_item(text: str) -> bool:
    """Return True when the text looks like a numbered list item."""
    return extract_numbered_list_index(text) is not None


def find_numbered_list_runs(sentences: Sequence[str]) -> List[Tuple[int, int]]:
    """Return (start, end) sentence ranges for numbered list runs."""
    runs: List[Tuple[int, int]] = []
    run_start: int | None = None
    last_number: int | None = None

    for idx, sentence in enumerate(sentences):
        number = extract_numbered_list_index(sentence)
        if number is None:
            if run_start is not None:
                runs.append((run_start, idx))
                run_start = None
                last_number = None
            continue

        if run_start is None:
            run_start = idx
            last_number = number
            continue

        if last_number is not None and number == last_number + 1:
            last_number = number
            continue

        runs.append((run_start, idx))
        run_start = idx
        last_number = number

    if run_start is not None:
        runs.append((run_start, len(sentences)))

    return [(start, end) for start, end in runs if end - start >= 2]


def build_chunk_prefix(
    *,
    document_ref: str | None,
    doc_type: str | None,
    section_path: Sequence[str] | None = None,
    chunk_position: str | None = None,
    list_header: str | None = None,
    max_length: int = 200,
) -> str:
    """Build a compact prefix for chunk text to improve retrieval."""
    parts: list[str] = []
    if document_ref:
        ref_text = document_ref.strip()
        if ref_text:
            parts.append(ref_text)
    if doc_type:
        doc_type_text = doc_type.strip()
        if doc_type_text:
            if parts:
                parts[-1] = f"{parts[-1]} - {doc_type_text}"
            else:
                parts.append(doc_type_text)
    path_parts: list[str] = []
    if section_path:
        path_parts.extend(
            segment.strip() for segment in section_path if segment and segment.strip()
        )
    if list_header:
        header_text = list_header.strip()
        if header_text:
            path_parts.append(f"Liste: {header_text}")
    if path_parts:
        parts.append(" > ".join(path_parts))
    if chunk_position:
        position_text = chunk_position.strip()
        if position_text:
            parts.append(position_text)

    if not parts:
        return ""

    prefix = " | ".join(parts)
    if max_length > 0 and len(prefix) > max_length:
        prefix = prefix[: max_length - 3].rstrip() + "..."
    return f"{prefix}\n"


__all__ = [
    "split_sentences",
    "extract_numbered_list_index",
    "is_numbered_list_item",
    "find_numbered_list_runs",
    "build_chunk_prefix",
]
