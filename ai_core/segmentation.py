from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    from markdown_it import MarkdownIt  # type: ignore
    from markdown_it.token import Token  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MarkdownIt = None  # type: ignore
    Token = object  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from markdown_it.token import Token as MarkdownToken  # type: ignore
else:
    MarkdownToken = Token  # type: ignore

if MarkdownIt is not None:  # pragma: no branch - simple guard
    _MARKDOWN = MarkdownIt("commonmark")
else:  # pragma: no cover - fallback path
    _MARKDOWN = None


_BLOCK_TOKEN_TYPES = {
    "heading_open": "heading",
    "paragraph_open": "paragraph",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "fence": "code",
    "code_block": "code",
    "table_open": "table",
}

_LIST_TOKENS = {"bullet_list_open", "ordered_list_open"}


def _slice_lines(lines: List[str], token: MarkdownToken) -> str:
    if token.map is None:
        return ""
    start, end = token.map
    if start >= end:
        return ""
    start = max(0, start)
    end = min(len(lines), end)
    block = "\n".join(lines[start:end])
    return block.strip("\n")


_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+.*")
_LIST_PATTERN = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+).*")
_TABLE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
_CODE_FENCE_PATTERN = re.compile(r"^\s*```+")


def _fallback_segment(text: str) -> List[str]:
    lines = text.splitlines()
    segments: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            segment = "\n".join(buffer).strip("\n")
            if segment:
                segments.append(segment)
            buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if _CODE_FENCE_PATTERN.match(line):
            flush()
            fence_lines = [line]
            i += 1
            while i < len(lines):
                fence_lines.append(lines[i])
                if _CODE_FENCE_PATTERN.match(lines[i]):
                    i += 1
                    break
                i += 1
            segments.append("\n".join(fence_lines).strip("\n"))
            continue
        if _HEADING_PATTERN.match(line):
            flush()
            segments.append(line.strip())
            i += 1
            continue
        if _TABLE_PATTERN.match(line):
            flush()
            table_lines = [line]
            i += 1
            while i < len(lines) and (
                _TABLE_PATTERN.match(lines[i])
                or set(lines[i].strip()) <= {":", "-", "|", " ", "+"}
            ):
                table_lines.append(lines[i])
                i += 1
            segments.append("\n".join(table_lines).strip("\n"))
            continue
        if _LIST_PATTERN.match(line):
            flush()
            list_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if _LIST_PATTERN.match(next_line):
                    list_lines.append(next_line)
                    i += 1
                    continue
                if next_line.strip().startswith("    ") or next_line.startswith("    "):
                    list_lines.append(next_line)
                    i += 1
                    continue
                break
            segments.append("\n".join(list_lines).strip("\n"))
            continue
        buffer.append(line)
        i += 1

    flush()
    return segments


def _markdown_segment(text: str) -> List[str]:
    lines = text.splitlines()
    tokens = _MARKDOWN.parse(text)
    segments: List[str] = []
    stack: List[MarkdownToken] = []

    for token in tokens:
        if token.nesting == 1:  # opening token
            stack.append(token)
            if token.type not in _BLOCK_TOKEN_TYPES or token.map is None:
                continue
            if token.type == "paragraph_open" and any(
                ancestor.type in _LIST_TOKENS for ancestor in stack[:-1]
            ):
                continue
            block = _slice_lines(lines, token)
            if block:
                segments.append(block)
        elif token.nesting == 0 and token.type in {"fence", "code_block"}:
            block = _slice_lines(lines, token)
            if block:
                segments.append(block)
        elif token.nesting == -1:
            if stack:
                stack.pop()

    return [segment for segment in segments if segment.strip()]


def segment_markdown_blocks(text: str) -> List[str]:
    """Return a list of structural Markdown blocks in document order."""

    stripped = text.strip("\n")
    if not stripped:
        return []

    if _MARKDOWN is not None:
        segments = _markdown_segment(text)
    else:  # pragma: no cover - exercised in environments without markdown-it
        segments = _fallback_segment(text)

    if not segments and stripped:
        return [stripped]

    return segments
