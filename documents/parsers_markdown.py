"""Markdown parser implementation producing normalized text blocks and assets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple
from urllib.parse import urlparse

import yaml
from lxml import html
from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin

from dateutil import parser as date_parser

from documents.contract_utils import (
    is_bcp47_like,
    normalize_media_type,
    normalize_optional_string,
    normalize_string,
    truncate_text,
)
from documents.parsers import DocumentParser, ParsedAsset, ParsedResult, ParsedTextBlock


_MARKDOWN_MEDIA_TYPES = {"text/markdown", "text/x-markdown"}
_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd", ".mkdn", ".mdtxt"}
_MAX_BLOCK_BYTES = 8 * 1024
_TABLE_PREVIEW_LIMIT = 500
_CONTEXT_LIMIT = 512
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:\s*(.*)$")
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]]+)\]")
_MULTI_SPACE_RE = re.compile(r"\s+")

_IMAGE_EXTENSION_FALLBACK = {
    ".apng": "image/apng",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def _normalise_media_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return normalize_media_type(value)
    except ValueError:
        return None


def _extract_candidate(obj: Any, key: str) -> Optional[str]:
    if hasattr(obj, key):
        value = getattr(obj, key)
        if isinstance(value, str):
            return value
    if isinstance(obj, Mapping):
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return None


def _extract_media_type(document: Any) -> Optional[str]:
    media_type = _extract_candidate(document, "media_type")
    return _normalise_media_type(media_type)


def _extract_blob(document: Any) -> Any:
    blob = getattr(document, "blob", None)
    if blob is None and isinstance(document, Mapping):
        blob = document.get("blob")
    return blob


def _decode_blob_payload(blob: Any) -> Optional[bytes]:
    if blob is None:
        return None
    if hasattr(blob, "decoded_payload"):
        payload = blob.decoded_payload()
        if isinstance(payload, bytes):
            return payload
    data = _extract_candidate(blob, "content")
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(blob, Mapping):
        base64_data = blob.get("base64")
        if isinstance(base64_data, (bytes, bytearray)):
            return bytes(base64_data)
        if isinstance(base64_data, str):
            try:
                import base64

                return base64.b64decode(base64_data)
            except Exception:  # pragma: no cover - malformed payloads fall back to None
                return None
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    return None


def _extract_filename(document: Any) -> Optional[str]:
    candidates = ("file_name", "filename", "name", "path")
    for key in candidates:
        value = _extract_candidate(document, key)
        if value:
            return value
    meta = getattr(document, "meta", None)
    if meta is not None:
        origin = _extract_candidate(meta, "origin_uri")
        if origin:
            return origin
    if isinstance(document, Mapping):
        meta_mapping = document.get("meta")
        if meta_mapping is not None:
            origin = _extract_candidate(meta_mapping, "origin_uri")
            if origin:
                return origin
    return None


def _decode_text(payload: Optional[bytes]) -> str:
    if not payload:
        return ""
    return payload.decode("utf-8", errors="replace")


def _strip_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines:
        return {}, text
    terminator = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            terminator = index
            break
    if terminator is None:
        return {}, text
    front_matter_lines = lines[1:terminator]
    rest_lines = lines[terminator + 1 :]
    front_matter_raw = "\n".join(front_matter_lines)
    try:
        parsed = yaml.safe_load(front_matter_raw) or {}
        if not isinstance(parsed, Mapping):
            parsed = {}
    except yaml.YAMLError:
        parsed = {}
    body = "\n".join(rest_lines)
    return dict(parsed), body


def _normalise_front_matter(data: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    title = normalize_optional_string(data.get("title"))
    if title:
        result["front_matter.title"] = title
    author = normalize_optional_string(data.get("author"))
    if author:
        result["front_matter.author"] = author
    language = normalize_optional_string(data.get("language"))
    if language and is_bcp47_like(language):
        result["front_matter.language"] = language
    raw_tags = data.get("tags")
    tags: Iterable[str]
    if isinstance(raw_tags, str):
        candidate = normalize_string(raw_tags)
        tags = [candidate.lower()] if candidate else []
    elif isinstance(raw_tags, Iterable):
        tags = [normalize_string(str(tag)).lower() for tag in raw_tags]
    else:
        tags = []
    tags = [tag for tag in tags if tag]
    if tags:
        unique_tags = list(dict.fromkeys(tags))
        result["front_matter.tags"] = unique_tags
    raw_date = data.get("date")
    date_value: Optional[str] = None
    if isinstance(raw_date, datetime):
        dt = raw_date
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        date_value = dt.isoformat()
    elif isinstance(raw_date, date):
        date_value = raw_date.isoformat()
    else:
        candidate = normalize_optional_string(raw_date)
        if candidate:
            try:
                parsed = date_parser.parse(candidate)
            except (ValueError, OverflowError):
                parsed = None
            if isinstance(parsed, datetime):
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc)
                date_value = parsed.isoformat()
            elif isinstance(parsed, date):
                date_value = parsed.isoformat()
            else:
                date_value = candidate
    if date_value:
        result["front_matter.date"] = date_value
    return result


def _extract_footnotes(text: str) -> Tuple[Dict[str, str], str]:
    footnotes: Dict[str, str] = {}
    output_lines: List[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _FOOTNOTE_DEF_RE.match(line)
        if match:
            label = match.group(1)
            value = [match.group(2).strip()]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate.startswith("    ") or candidate.startswith("\t"):
                    value.append(candidate.strip())
                    index += 1
                else:
                    break
            normalised = normalize_string(" ".join(part for part in value if part))
            if normalised:
                footnotes[label] = normalised
            continue
        output_lines.append(line)
        index += 1
    return footnotes, "\n".join(output_lines)


def _collapse_whitespace(value: str) -> str:
    value = value.replace("\t", " ").replace("\r", " ")
    value = normalize_string(value)
    return _MULTI_SPACE_RE.sub(" ", value).strip()


def _apply_footnotes(text: str, footnotes: Mapping[str, str]) -> str:
    if not footnotes:
        return text
    references: List[str] = []

    def _collect(match: re.Match[str]) -> str:
        label = match.group(1)
        references.append(label)
        return ""

    stripped = _FOOTNOTE_REF_RE.sub(_collect, text)
    suffix_parts = []
    seen: MutableSequence[str] = []
    for label in references:
        if label in seen:
            continue
        seen.append(label)
        value = footnotes.get(label)
        if value:
            suffix_parts.append(f"[fn: {value}]")
    suffix = " ".join(suffix_parts)
    stripped = stripped.strip()
    if suffix:
        return f"{stripped} {suffix}" if stripped else suffix
    return stripped


def _truncate_block_text(text: str) -> str:
    truncated = truncate_text(text, _MAX_BLOCK_BYTES)
    return truncated or ""


def _context_snippet(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    snippet = truncate_text(text, _CONTEXT_LIMIT)
    return snippet or None


def _infer_media_type(file_uri: str) -> str:
    if file_uri.startswith("data:"):
        try:
            prefix = file_uri.split(";", 1)[0]
            return normalize_media_type(prefix.split(":", 1)[1])
        except (IndexError, ValueError):
            return "image/*"
    parsed = urlparse(file_uri)
    path = parsed.path or file_uri
    _, extension = os.path.splitext(path.lower())
    if extension and extension in _IMAGE_EXTENSION_FALLBACK:
        return _IMAGE_EXTENSION_FALLBACK[extension]
    return "image/*"


def _normalise_code_language(info: str) -> Optional[str]:
    cleaned = normalize_optional_string(info)
    if not cleaned:
        return None
    primary = cleaned.split()[0]
    return primary or None


@dataclass
class _AssetDraft:
    file_uri: str
    media_type: str
    context_before: Optional[str]
    context_after: Optional[str]
    alt_text: Optional[str] = None


class _MarkdownState:
    def __init__(
        self,
        *,
        default_language: Optional[str],
        origin_uri: Optional[str],
        footnotes: Mapping[str, str],
    ) -> None:
        self.default_language = default_language
        self.origin_uri = origin_uri
        self.footnotes = footnotes
        self.section_stack: List[str] = []
        self.blocks: List[ParsedTextBlock] = []
        self.assets: List[_AssetDraft] = []
        self._pending_assets: List[_AssetDraft] = []
        self._last_text: Optional[str] = None
        self.word_count = 0
        self.heading_count = 0
        self.paragraph_count = 0
        self.list_count = 0
        self.table_count = 0
        self.code_count = 0

    def _section_path(self) -> Optional[Tuple[str, ...]]:
        return tuple(self.section_stack) if self.section_stack else None

    def push_heading(self, level: int, title: str) -> None:
        cleaned = _collapse_whitespace(title) if title else ""
        cleaned = cleaned or title.strip()
        while len(self.section_stack) >= level:
            self.section_stack.pop()
        if cleaned:
            self.section_stack.append(cleaned)
        else:
            self.section_stack.append(f"Heading {level}")

    def _update_pending_assets(self, context: Optional[str]) -> None:
        if not context:
            return
        snippet = _context_snippet(context)
        if not snippet:
            return
        for asset in list(self._pending_assets):
            if not asset.context_after:
                asset.context_after = snippet
        self._pending_assets.clear()

    def _apply_footnotes_to_text(self, text: str) -> str:
        return _apply_footnotes(text, self.footnotes)

    def add_text_block(
        self,
        *,
        text: str,
        kind: str,
        table_meta: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        if kind == "code":
            cleaned = truncate_text(text.strip("\n"), _MAX_BLOCK_BYTES) or ""
            language = None
        else:
            cleaned = _collapse_whitespace(text)
            cleaned = self._apply_footnotes_to_text(cleaned)
            cleaned = _truncate_block_text(cleaned)
            language = self.default_language
        if not cleaned:
            return None
        block = ParsedTextBlock(
            text=cleaned,
            kind=kind,
            section_path=self._section_path(),
            table_meta=table_meta,
            language=language,
        )
        self.blocks.append(block)
        if kind == "heading":
            self.heading_count += 1
        elif kind == "paragraph":
            self.paragraph_count += 1
        elif kind == "list":
            self.list_count += 1
        elif kind == "table_summary":
            self.table_count += 1
        elif kind == "code":
            self.code_count += 1
        if kind != "code":
            self.word_count += len(block.text.split())
        self._last_text = block.text
        self._update_pending_assets(block.text)
        return block.text

    def add_asset(
        self,
        *,
        file_uri: str,
        before_text: Optional[str],
        after_text: Optional[str],
        alt_text: Optional[str],
    ) -> None:
        media_type = _infer_media_type(file_uri)
        before_context = _context_snippet(before_text or self._last_text)
        after_context = _context_snippet(after_text)
        asset = _AssetDraft(
            file_uri=file_uri,
            media_type=media_type,
            context_before=before_context,
            context_after=after_context,
            alt_text=alt_text,
        )
        if after_context is None:
            self._pending_assets.append(asset)
        self.assets.append(asset)

    def finalise_assets(self) -> List[ParsedAsset]:
        results: List[ParsedAsset] = []
        for asset in self.assets:
            context_after = asset.context_after
            if not context_after and asset.alt_text:
                context_after = _context_snippet(asset.alt_text)
            if context_after and self.origin_uri:
                combined = f"{context_after}\nSource: {self.origin_uri}"
                context_after = _context_snippet(combined)
            elif self.origin_uri:
                context_after = _context_snippet(f"Source: {self.origin_uri}")
            parsed_asset = ParsedAsset(
                media_type=asset.media_type,
                file_uri=asset.file_uri,
                context_before=asset.context_before,
                context_after=context_after,
            )
            results.append(parsed_asset)
        return results


class MarkdownDocumentParser(DocumentParser):
    """Parse GitHub-flavoured Markdown documents into structured results."""

    _markdown: Optional[MarkdownIt] = None

    def can_handle(self, document: Any) -> bool:
        media_type = _extract_media_type(document)
        if media_type in _MARKDOWN_MEDIA_TYPES:
            return True
        blob = _extract_blob(document)
        blob_media = _extract_media_type(blob)
        if blob_media in _MARKDOWN_MEDIA_TYPES:
            return True
        filename = _extract_filename(document)
        if filename:
            lower = filename.lower()
            if any(lower.endswith(ext) for ext in _MARKDOWN_EXTENSIONS):
                return True
        payload = _decode_blob_payload(blob)
        if payload and payload.lstrip().startswith(b"#"):
            return True
        return False

    @classmethod
    def _markdown_parser(cls) -> MarkdownIt:
        if cls._markdown is None:
            parser = MarkdownIt("commonmark", {"html": True, "linkify": False})
            parser.enable("table")
            parser.use(tasklists_plugin)
            cls._markdown = parser
        return cls._markdown

    def parse(self, document: Any, config: Any) -> ParsedResult:
        blob = _extract_blob(document)
        payload = _decode_blob_payload(blob)
        text = _decode_text(payload).lstrip("\ufeff")
        front_matter_raw, body = _strip_front_matter(text)
        footnotes, body = _extract_footnotes(body)
        front_matter_stats = _normalise_front_matter(front_matter_raw)
        default_language = front_matter_stats.get("front_matter.language")
        meta = getattr(document, "meta", None)
        if default_language is None and meta is not None:
            meta_language = _extract_candidate(meta, "language")
            if meta_language and is_bcp47_like(meta_language):
                default_language = meta_language
        origin_uri = _extract_candidate(meta, "origin_uri") if meta is not None else None
        state = _MarkdownState(
            default_language=default_language,
            origin_uri=origin_uri,
            footnotes=footnotes,
        )
        tokens = self._markdown_parser().parse(body)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.type == "heading_open":
                level = int(token.tag[1]) if token.tag and token.tag.startswith("h") else 1
                inline = tokens[index + 1] if index + 1 < len(tokens) else None
                heading_text = inline.content if inline and inline.type == "inline" else ""
                state.push_heading(level, heading_text)
                state.add_text_block(text=heading_text, kind="heading")
                index += 3
                continue
            if token.type == "paragraph_open":
                inline = tokens[index + 1] if index + 1 < len(tokens) else None
                text_value = self._render_inline(inline, state)
                state.add_text_block(text=text_value, kind="paragraph")
                index += 3
                continue
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                index = self._process_list(tokens, index, state)
                continue
            if token.type == "fence":
                language_hint = _normalise_code_language(token.info or "")
                table_meta = {"lang": language_hint} if language_hint else None
                state.add_text_block(text=token.content, kind="code", table_meta=table_meta)
                index += 1
                continue
            if token.type == "code_block":
                state.add_text_block(text=token.content, kind="code")
                index += 1
                continue
            if token.type == "table_open":
                index = self._process_table(tokens, index, state)
                continue
            if token.type in {"html_block", "html_inline"}:
                self._process_html_block(token.content, state)
                index += 1
                continue
            index += 1

        assets = state.finalise_assets()
        statistics: Dict[str, Any] = {
            "parser.words": state.word_count,
            "parser.headings": state.heading_count,
            "parser.paragraphs": state.paragraph_count,
            "parser.lists": state.list_count,
            "parser.tables": state.table_count,
            "assets.images": len(assets),
            "parser.code_blocks": state.code_count,
            "parser.front_matter": bool(front_matter_raw),
        }
        statistics.update(front_matter_stats)
        if origin_uri and assets:
            statistics["assets.origins"] = {
                asset.file_uri: origin_uri for asset in assets
            }
        return ParsedResult(
            text_blocks=tuple(state.blocks),
            assets=tuple(assets),
            statistics=statistics,
        )

    def _render_inline(self, inline_token: Optional[Any], state: _MarkdownState) -> str:
        if inline_token is None or inline_token.type != "inline":
            return ""
        children = inline_token.children or []
        text_fragments: List[str] = []
        positions: List[Tuple[int, Dict[str, str]]] = []
        for child in children:
            if child.type == "image":
                src = child.attrGet("src") or ""
                if not src:
                    continue
                alt_text = child.attrGet("alt") or child.content or ""
                before_text = "".join(text_fragments)
                positions.append((len(before_text), {"src": src, "alt": alt_text}))
            elif child.type in {
                "link_open",
                "link_close",
                "em_open",
                "em_close",
                "strong_open",
                "strong_close",
                "s_open",
                "s_close",
                "del_open",
                "del_close",
            }:
                continue
            elif child.type in {"softbreak", "hardbreak"}:
                text_fragments.append(" ")
            elif child.type == "code_inline":
                text_fragments.append(child.content)
            elif child.type == "html_inline":
                rendered = self._process_html_inline(
                    child.content, state, "".join(text_fragments)
                )
                if rendered:
                    text_fragments.append(rendered)
            else:
                text_fragments.append(child.content)
        full_parts: List[str] = []
        for fragment in text_fragments:
            if not fragment:
                continue
            if (
                full_parts
                and not full_parts[-1][-1].isspace()
                and not fragment[0].isspace()
            ):
                full_parts.append(" ")
            full_parts.append(fragment)
        full_text = "".join(full_parts)
        for offset, payload in positions:
            before = full_text[:offset]
            after = full_text[offset:]
            state.add_asset(
                file_uri=payload["src"],
                before_text=before,
                after_text=after,
                alt_text=payload["alt"],
            )
        return full_text

    def _process_list(
        self, tokens: Sequence[Any], index: int, state: _MarkdownState
    ) -> int:
        level = 1
        index += 1
        while index < len(tokens) and level > 0:
            token = tokens[index]
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                level += 1
                index += 1
                continue
            if token.type in {"bullet_list_close", "ordered_list_close"}:
                level -= 1
                index += 1
                continue
            if token.type == "list_item_open":
                text, index = self._collect_list_item(tokens, index, state)
                state.add_text_block(text=text, kind="list")
                continue
            index += 1
        return index

    def _collect_list_item(
        self, tokens: Sequence[Any], start: int, state: _MarkdownState
    ) -> Tuple[str, int]:
        index = start + 1
        fragments: List[str] = []
        while index < len(tokens):
            token = tokens[index]
            if token.type == "list_item_close":
                text = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
                return text, index + 1
            if token.type == "inline":
                fragments.append(self._render_inline(token, state))
                index += 1
                continue
            if token.type == "paragraph_open":
                inline = tokens[index + 1] if index + 1 < len(tokens) else None
                fragments.append(self._render_inline(inline, state))
                index += 3
                continue
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                index = self._process_list(tokens, index, state)
                continue
            index += 1
        text = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
        return text, index

    def _process_table(
        self, tokens: Sequence[Any], index: int, state: _MarkdownState
    ) -> int:
        headers: List[str] = []
        rows: List[List[str]] = []
        column_count = 0
        cursor = index + 1
        while cursor < len(tokens):
            token = tokens[cursor]
            if token.type == "tr_open":
                cursor += 1
                cells: List[str] = []
                cell_tags: List[str] = []
                while cursor < len(tokens):
                    cell_token = tokens[cursor]
                    if cell_token.type in {"th_open", "td_open"}:
                        cell_tags.append(cell_token.type)
                        inline = (
                            tokens[cursor + 1] if cursor + 1 < len(tokens) else None
                        )
                        content = (
                            inline.content if inline and inline.type == "inline" else ""
                        )
                        cells.append(_collapse_whitespace(content))
                        cursor += 3
                        continue
                    if cell_token.type == "tr_close":
                        cursor += 1
                        break
                    cursor += 1
                column_count = max(column_count, len(cells))
                if cells:
                    if not headers and all(tag == "th_open" for tag in cell_tags):
                        headers = cells
                    else:
                        rows.append(cells)
                continue
            if token.type == "table_close":
                index = cursor + 1
                break
            cursor += 1
        preview_rows = []
        if headers:
            preview_rows.append(headers)
        preview_rows.extend(rows[:5])
        preview_lines = [
            "\t".join(row) for row in preview_rows if any(cell for cell in row)
        ]
        preview_text = "\n".join(preview_lines)
        preview_text = truncate_text(preview_text, _TABLE_PREVIEW_LIMIT) or ""
        summary = f"Table rows={len(rows)} columns={column_count}"
        if preview_text:
            summary = f"{summary}\n{preview_text}"
        summary = truncate_text(summary, _TABLE_PREVIEW_LIMIT) or summary
        sample_rows = [row for row in rows[:5] if any(cell for cell in row)]
        table_meta = {
            "rows": len(rows),
            "columns": column_count,
            "headers": headers,
            "sample_row_count": len(sample_rows),
            "sample_rows": sample_rows,
        }
        state.add_text_block(text=summary, kind="table_summary", table_meta=table_meta)
        return index

    def _process_html_block(self, html_content: str, state: _MarkdownState) -> None:
        fragments = html.fragments_fromstring(html_content)
        for fragment in fragments:
            if isinstance(fragment, str):
                text = _collapse_whitespace(fragment)
                if text:
                    state.add_text_block(text=text, kind="paragraph")
                continue
            self._scan_html_for_assets(fragment, state, state._last_text)
            tag = (fragment.tag or "").lower()
            if tag in {"table"}:
                self._consume_html_table(fragment, state)
                continue
            if tag in {"ul", "ol"}:
                for li in fragment.findall(".//li"):
                    li_text = _collapse_whitespace(li.text_content())
                    if li_text:
                        state.add_text_block(text=li_text, kind="list")
                continue
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                text = _collapse_whitespace(fragment.text_content())
                level = int(tag[1]) if len(tag) > 1 and tag[1].isdigit() else 1
                state.push_heading(level, text)
                state.add_text_block(text=text, kind="heading")
                continue
            text = _collapse_whitespace(fragment.text_content())
            if text:
                state.add_text_block(text=text, kind="paragraph")

    def _process_html_inline(
        self, html_content: str, state: _MarkdownState, current_text: str
    ) -> str:
        fragments = html.fragments_fromstring(html_content)
        collected: List[str] = []
        for fragment in fragments:
            if isinstance(fragment, str):
                collected.append(fragment)
                continue
            self._scan_html_for_assets(fragment, state, current_text)
            text = _collapse_whitespace(fragment.text_content())
            if text:
                collected.append(text)
        return " ".join(part.strip() for part in collected if part.strip())

    def _scan_html_for_assets(
        self, element: html.HtmlElement, state: _MarkdownState, before_text: Optional[str]
    ) -> None:
        for img in element.findall(".//img"):
            src = img.get("src")
            if not src:
                continue
            alt_text = img.get("alt") or img.get("title") or ""
            after_text = ""
            parent = img.getparent()
            if parent is not None and parent.tag and parent.tag.lower() == "figure":
                caption = parent.find("figcaption")
                if caption is not None:
                    after_text = caption.text_content()
            if not after_text:
                after_text = img.tail or ""
            state.add_asset(
                file_uri=src,
                before_text=before_text,
                after_text=_collapse_whitespace(after_text) if after_text else after_text,
                alt_text=alt_text,
            )

    def _consume_html_table(self, element: html.HtmlElement, state: _MarkdownState) -> None:
        headers: List[str] = []
        rows: List[List[str]] = []
        column_count = 0
        for tr in element.findall(".//tr"):
            cells: List[str] = []
            header_row = True
            for cell in list(tr):
                if cell.tag is None:
                    continue
                tag = cell.tag.lower()
                if tag not in {"th", "td"}:
                    continue
                if tag != "th":
                    header_row = False
                cells.append(_collapse_whitespace(cell.text_content()))
            if not cells:
                continue
            column_count = max(column_count, len(cells))
            if header_row and not headers:
                headers = cells
            else:
                rows.append(cells)
        preview_rows = []
        if headers:
            preview_rows.append(headers)
        preview_rows.extend(rows[:5])
        preview_lines = [
            "\t".join(row) for row in preview_rows if any(cell for cell in row)
        ]
        preview_text = "\n".join(preview_lines)
        preview_text = truncate_text(preview_text, _TABLE_PREVIEW_LIMIT) or ""
        summary = f"Table rows={len(rows)} columns={column_count}"
        if preview_text:
            summary = f"{summary}\n{preview_text}"
        summary = truncate_text(summary, _TABLE_PREVIEW_LIMIT) or summary
        sample_rows = [row for row in rows[:5] if any(cell for cell in row)]
        table_meta = {
            "rows": len(rows),
            "columns": column_count,
            "headers": headers,
            "sample_row_count": len(sample_rows),
            "sample_rows": sample_rows,
        }
        state.add_text_block(text=summary, kind="table_summary", table_meta=table_meta)

