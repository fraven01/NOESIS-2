"""HTML parser implementation extracting reader content and assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lxml import html
from lxml.etree import ParserError

from documents.contract_utils import (
    is_bcp47_like,
    normalize_media_type,
    normalize_optional_string,
    normalize_string,
    truncate_text,
)
from documents.media_type_resolver import resolve_image_media_type
from documents.parsers import (
    DocumentParser,
    ParsedAsset,
    ParsedResult,
    ParsedTextBlock,
    build_parsed_asset_with_meta,
    build_parsed_result,
    build_parsed_text_block,
)
from documents.payloads import extract_payload


_HTML_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}
_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_TABLE_PREVIEW_LIMIT = 500
_CONTEXT_LIMIT = 512
_MAX_BLOCK_BYTES = 8 * 1024


def _config_flag(config: Any, name: str, default: bool) -> bool:
    if config is None:
        return default
    if hasattr(config, name):
        value = getattr(config, name)
    elif isinstance(config, Mapping):
        value = config.get(name, default)  # type: ignore[index]
    else:
        return default
    if isinstance(value, bool):
        return value
    return bool(value)


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
    if not media_type:
        # Check metadata pipeline_config for explicit override
        if hasattr(document, "meta") and hasattr(document.meta, "pipeline_config"):
            cfg = document.meta.pipeline_config
            if cfg and isinstance(cfg, Mapping):
                media_type = cfg.get("media_type")
    if not media_type:
        return None
    try:
        return normalize_media_type(media_type)
    except ValueError:
        return None


def _extract_blob(document: Any) -> Any:
    blob = getattr(document, "blob", None)
    if blob is None and isinstance(document, Mapping):
        blob = document.get("blob")
    return blob


def _decode_blob_payload(blob: Any) -> Optional[bytes]:
    if blob is None:
        return None
    encoding = None
    if hasattr(blob, "content_encoding"):
        candidate = getattr(blob, "content_encoding")
        if isinstance(candidate, str):
            encoding = candidate
    return extract_payload(blob, content_encoding=encoding)


def _decode_text(payload: Optional[bytes]) -> str:
    if not payload:
        return ""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("utf-8", errors="replace")


def _collapse_whitespace(value: str) -> str:
    normalized = normalize_string(value)
    return " ".join(part for part in normalized.split() if part)


def _truncate_block_text(text: str) -> str:
    truncated = truncate_text(text, _MAX_BLOCK_BYTES)
    return truncated or ""


def _context_snippet(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    snippet = truncate_text(text, _CONTEXT_LIMIT)
    return snippet or None


@dataclass
class _AssetDraft:
    file_uri: str
    media_type: str
    context_before: Optional[str]
    context_after: Optional[str]
    alt_text: Optional[str] = None
    parent_ref: Optional[str] = None
    locator: Optional[str] = None


class _HtmlState:
    def __init__(
        self,
        *,
        default_language: Optional[str],
        origin_uri: Optional[str],
    ) -> None:
        self.default_language = default_language
        self.origin_uri = origin_uri
        self.section_stack: List[str] = []
        self.blocks: List[ParsedTextBlock] = []
        self.assets: List[_AssetDraft] = []
        self._pending_assets: List[_AssetDraft] = []
        self._last_text: Optional[str] = None
        self._asset_counter = 0
        self.word_count = 0
        self.heading_count = 0
        self.paragraph_count = 0
        self.list_count = 0
        self.table_count = 0

    def _section_path(self) -> Optional[Tuple[str, ...]]:
        return tuple(self.section_stack) if self.section_stack else None

    def push_heading(self, level: int, title: str) -> None:
        cleaned = _collapse_whitespace(title) if title else ""
        cleaned = cleaned or f"Heading {level}"
        while len(self.section_stack) >= level:
            self.section_stack.pop()
        # Truncate to 128 chars to comply with section_path validation
        self.section_stack.append(cleaned[:128])

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

    def add_text_block(
        self,
        *,
        text: str,
        kind: str,
        table_meta: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        if kind == "code":
            cleaned = truncate_text(text.strip("\n"), _MAX_BLOCK_BYTES) or ""
            language = self.default_language
        else:
            cleaned = _collapse_whitespace(text)
            cleaned = _truncate_block_text(cleaned)
            language = self.default_language
        if not cleaned:
            return None
        block = build_parsed_text_block(
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
        declared_media_type: Optional[str] = None,
    ) -> None:
        media_type = resolve_image_media_type(
            file_uri, declared_type=normalize_optional_string(declared_media_type)
        )
        before_context = _context_snippet(before_text or self._last_text)
        after_context = _context_snippet(after_text)
        self._asset_counter += 1
        if self.section_stack:
            parent_ref = f"section:{'>'.join(self.section_stack)}"
        else:
            parent_ref = "section:root"
        locator = f"{parent_ref}:asset:{self._asset_counter}"
        asset = _AssetDraft(
            file_uri=file_uri,
            media_type=media_type,
            context_before=before_context,
            context_after=after_context,
            alt_text=alt_text,
            parent_ref=parent_ref,
            locator=locator,
        )
        if after_context is None:
            self._pending_assets.append(asset)
        self.assets.append(asset)

    def finalise_assets(self) -> List[ParsedAsset]:
        results: List[ParsedAsset] = []
        for asset in self.assets:
            context_after = asset.context_after
            base_context_after = context_after
            if not context_after and asset.alt_text:
                base_context_after = _context_snippet(asset.alt_text)
                context_after = base_context_after
            if context_after and self.origin_uri:
                combined = f"{context_after}\nSource: {self.origin_uri}"
                context_after = _context_snippet(combined)
            elif self.origin_uri:
                context_after = _context_snippet(f"Source: {self.origin_uri}")
            metadata: Dict[str, Any] = {}
            candidates: List[Tuple[str, str]] = []
            if asset.alt_text:
                alt_candidate = _context_snippet(asset.alt_text) or asset.alt_text
                candidates.append(("alt_text", alt_candidate))
            if base_context_after:
                candidates.append(("context_after", base_context_after))
            if asset.context_before:
                candidates.append(("context_before", asset.context_before))
            if self.origin_uri:
                candidates.append(("origin", self.origin_uri))
                metadata["origin_uri"] = self.origin_uri
            if candidates:
                metadata["caption_candidates"] = candidates
            if asset.parent_ref:
                metadata["parent_ref"] = asset.parent_ref
            if asset.locator:
                metadata["locator"] = asset.locator
            parsed_asset = build_parsed_asset_with_meta(
                media_type=asset.media_type,
                file_uri=asset.file_uri,
                context_before=asset.context_before,
                context_after=context_after,
                metadata=metadata,
            )
            results.append(parsed_asset)
        return results


def _strip_boilerplate(root: html.HtmlElement) -> None:
    drop_tags = {
        "script",
        "style",
        "noscript",
        "template",
        "iframe",
        "object",
        "embed",
        "svg",
        "canvas",
    }
    for tag in drop_tags:
        for node in list(root.findall(f".//{tag}")):
            node.drop_tree()
    for node in list(
        root.xpath(
            ".//*[self::header or self::footer or self::aside or self::nav or @role='navigation' or @role='banner' or @role='contentinfo']"
        )
    ):
        if node is root:
            continue
        node.drop_tree()
    for node in list(
        root.xpath(
            ".//*[(contains(@class, 'nav') or contains(@class, 'menu') or contains(@class, 'sidebar') or contains(@class, 'footer') or contains(@id, 'nav') or contains(@id, 'menu') or contains(@id, 'sidebar') or contains(@id, 'footer'))]"
        )
    ):
        if node is root:
            continue
        node.drop_tree()


def _select_content_root(tree: html.HtmlElement) -> html.HtmlElement:
    candidates = []
    candidates.extend(tree.xpath("//main"))
    candidates.extend(tree.xpath("//article"))
    candidates.extend(tree.xpath("//*[@role='main']"))
    candidates.extend(
        tree.xpath(
            "//div[contains(@class, 'content') or contains(@class, 'article') or contains(@class, 'post') or contains(@id, 'content') or contains(@id, 'article') or contains(@id, 'post')]"
        )
    )
    for candidate in candidates:
        text_length = len("".join(candidate.itertext()).strip())
        if text_length > 200:
            return candidate
    body = tree.find("body")
    if body is not None:
        return body
    return tree


def _strip_placeholders(text: str, placeholders: Sequence[str]) -> str:
    for placeholder in placeholders:
        text = text.replace(placeholder, " ")
    return text


def _render_block(
    node: html.HtmlElement,
    state: _HtmlState,
    *,
    allow_inline_assets: bool = True,
) -> str:
    pieces: List[str] = []
    assets: List[Tuple[str, str, Optional[str]]] = []

    def walk(element: html.HtmlElement) -> None:
        text_value = element.text or ""
        if text_value:
            pieces.append(text_value)
        for child in element:
            tag = child.tag.lower() if isinstance(child.tag, str) else ""
            if tag in {"script", "style", "noscript", "template"}:
                if child.tail:
                    pieces.append(child.tail)
                continue
            if tag == "img" and allow_inline_assets:
                src = normalize_optional_string(child.get("src"))
                if src:
                    placeholder = f"[[[IMG{len(assets)}]]]"
                    pieces.append(placeholder)
                    alt_text = normalize_optional_string(child.get("alt"))
                    assets.append((placeholder, src, alt_text))
                if child.tail:
                    pieces.append(child.tail)
                continue
            if tag == "figure":
                if child.tail:
                    pieces.append(child.tail)
                continue
            if tag == "br":
                pieces.append(" ")
                if child.tail:
                    pieces.append(child.tail)
                continue
            walk(child)
            if child.tail:
                pieces.append(child.tail)

    walk(node)
    raw_text = "".join(pieces)
    if not raw_text.strip():
        return ""
    for placeholder, src, alt_text in assets:
        parts = raw_text.split(placeholder, 1)
        if len(parts) != 2:
            continue
        before_raw, after_raw = parts
        before_clean = _collapse_whitespace(
            _strip_placeholders(before_raw, [p for p, _, _ in assets])
        )
        after_clean = _collapse_whitespace(
            _strip_placeholders(
                after_raw, [p for p, _, _ in assets if p != placeholder]
            )
        )
        state.add_asset(
            file_uri=src,
            before_text=before_clean,
            after_text=after_clean,
            alt_text=alt_text,
        )
        raw_text = raw_text.replace(placeholder, " ")
    return raw_text


def _summarise_table(
    node: html.HtmlElement,
) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    rows: List[List[str]] = []
    for tr in node.xpath(".//tr"):
        cells = tr.xpath("./th|./td")
        values: List[str] = []
        for cell in cells:
            text = _collapse_whitespace("".join(cell.itertext()))
            if text:
                values.append(text)
            else:
                values.append("")
        if values:
            rows.append(values)
    if not rows:
        return None, None
    header_rows = node.xpath(".//thead//tr")
    header_candidates = node.xpath(".//thead//th") or node.xpath(".//tr[1]//th")
    if header_candidates:
        headers = [
            _collapse_whitespace("".join(cell.itertext())) for cell in header_candidates
        ]
        header_count = len(header_rows) if header_rows else 1
        data_rows = rows[header_count:]
    else:
        headers = rows[0]
        data_rows = rows[1:]
    data_rows = data_rows or []
    column_count = max(len(row) for row in ([headers] + data_rows) if row)
    headers = headers[:column_count]
    sample_rows = [
        row[:column_count] for row in data_rows[:5] if any(cell for cell in row)
    ]
    preview_rows: List[List[str]] = []
    if headers:
        preview_rows.append(headers)
    preview_rows.extend(sample_rows)
    preview_lines = [
        "\t".join(row) for row in preview_rows if any(cell for cell in row)
    ]
    preview_text = "\n".join(preview_lines)
    preview_text = truncate_text(preview_text, _TABLE_PREVIEW_LIMIT) or ""
    summary = f"Table rows={len(data_rows)} columns={column_count}"
    if preview_text:
        summary = f"{summary}\n{preview_text}"
    summary = truncate_text(summary, _TABLE_PREVIEW_LIMIT) or summary
    table_meta: Dict[str, Any] = {
        "rows": len(data_rows),
        "columns": column_count,
        "headers": headers,
        "sample_row_count": len(sample_rows),
    }
    if sample_rows:
        table_meta["sample_rows"] = sample_rows
    return summary, table_meta


def _collect_language(tree: html.HtmlElement, document: Any) -> Optional[str]:
    html_element = tree if tree.tag == "html" else None
    if html_element is None:
        matches = tree.xpath("//html")
        html_element = matches[0] if matches else None
    lang_candidates: Iterable[str] = []
    if html_element is not None:
        lang_values = [html_element.get("lang"), html_element.get("xml:lang")]
        lang_candidates = [value for value in lang_values if value]
    else:
        lang_candidates = []
    meta_langs = tree.xpath("//meta[@http-equiv='content-language']/@content")
    all_candidates = list(lang_candidates) + [value for value in meta_langs if value]
    if hasattr(document, "meta"):
        meta_language = _extract_candidate(document.meta, "language")
        if meta_language:
            all_candidates.append(meta_language)
    for candidate in all_candidates:
        normalized = normalize_optional_string(candidate)
        if normalized and is_bcp47_like(normalized):
            return normalized
    return None


def _collect_origin(document: Any) -> Optional[str]:
    meta = getattr(document, "meta", None)
    if meta is not None:
        origin = _extract_candidate(meta, "origin_uri")
        if origin:
            return origin
    if isinstance(document, Mapping):
        meta_mapping = document.get("meta")
        if isinstance(meta_mapping, Mapping):
            origin = _extract_candidate(meta_mapping, "origin_uri")
            if origin:
                return origin
    return None


class HtmlDocumentParser(DocumentParser):
    """Parse HTML documents into structured reader content."""

    def can_handle(self, document: Any) -> bool:
        media_type = _extract_media_type(document)
        if media_type in _HTML_MEDIA_TYPES:
            return True
        blob = _extract_blob(document)
        blob_media = _extract_media_type(blob)
        if blob_media in _HTML_MEDIA_TYPES:
            return True
        filename = _extract_candidate(document, "file_name") or _extract_candidate(
            document, "filename"
        )
        if not filename and isinstance(document, Mapping):
            filename = document.get("name")
        if filename:
            lower = filename.lower()
            if any(lower.endswith(ext) for ext in _HTML_EXTENSIONS):
                return True
        payload = _decode_blob_payload(blob)
        if payload:
            prefix = payload.lstrip()[:20].lower()
            if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
                return True
        return False

    def parse(self, document: Any, config: Any) -> ParsedResult:
        from documents.contracts import LocalFileBlob

        text = ""
        blob = getattr(document, "blob", None)

        # Prefer local file path if available (Download Middleware Pattern)
        if isinstance(blob, LocalFileBlob):
            try:
                with open(blob.path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as exc:
                raise ValueError(f"html_parse_failed_file_read: {exc}") from exc
        else:
            # Fallback to blob extraction/decoding
            blob = _extract_blob(document)
            payload = _decode_blob_payload(blob)
            text = _decode_text(payload).lstrip("\ufeff")

        try:
            tree = html.fromstring(text)
        except (ParserError, TypeError) as exc:
            raise ValueError("html_parse_failed") from exc
        use_readability = _config_flag(config, "use_readability_html_extraction", False)
        content_root = None
        if use_readability:
            content_root = self._extract_main_content_with_readability(text)
        if content_root is None:
            content_root = _select_content_root(tree)
        _strip_boilerplate(content_root)
        default_language = _collect_language(tree, document)
        origin_uri = _collect_origin(document)
        state = _HtmlState(default_language=default_language, origin_uri=origin_uri)
        self._process_element(content_root, state)
        assets = state.finalise_assets()
        statistics: Dict[str, Any] = {
            "parser.words": state.word_count,
            "parser.headings": state.heading_count,
            "parser.paragraphs": state.paragraph_count,
            "parser.lists": state.list_count,
            "parser.tables": state.table_count,
            "assets.images": len(assets),
        }
        if origin_uri and assets:
            statistics["assets.origins"] = {
                asset.file_uri: origin_uri for asset in assets
            }
        return build_parsed_result(
            text_blocks=tuple(state.blocks),
            assets=tuple(assets),
            statistics=statistics,
        )

    def _extract_main_content_with_readability(
        self, raw_html: str
    ) -> Optional[html.HtmlElement]:
        try:
            from readability import Document as ReadabilityDocument
        except ImportError:  # pragma: no cover - optional dependency guard
            return None
        try:
            reader_document = ReadabilityDocument(raw_html)
            summary_fragment = reader_document.summary(html_partial=True)
        except Exception:  # pragma: no cover - defensive guard around parser failures
            return None
        if not summary_fragment:
            return None
        try:
            fragment_root = html.fromstring(summary_fragment)
        except (ParserError, TypeError):
            return None
        for xpath in (".//article", ".//main", ".//*[@role='main']"):
            matches = fragment_root.xpath(xpath)
            for match in matches:
                if not isinstance(getattr(match, "tag", None), str):
                    continue
                text_length = len("".join(match.itertext()).strip())
                if text_length:
                    return match
        if isinstance(fragment_root.tag, str) and fragment_root.tag.lower() == "html":
            body = fragment_root.find("body")
            if body is not None:
                return body
        return fragment_root

    def _process_element(self, element: html.HtmlElement, state: _HtmlState) -> None:
        if not isinstance(element.tag, str):
            for child in element:
                self._process_element(child, state)
            return
        tag = element.tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            text = _collapse_whitespace("".join(element.itertext()))
            if text:
                state.push_heading(level, text)
                state.add_text_block(text=text, kind="heading")
            for child in element:
                self._process_element(child, state)
            return
        if tag == "p":
            raw_text = _render_block(element, state)
            state.add_text_block(text=raw_text, kind="paragraph")
            return
        if tag in {"ul", "ol"}:
            items: List[str] = []
            for li in element.xpath("./li"):
                text = _render_block(li, state)
                cleaned = _collapse_whitespace(text)
                if cleaned:
                    items.append(cleaned)
            if items:
                state.add_text_block(text="\n".join(items), kind="list")
            for li in element.xpath("./li"):
                for figure in li.xpath(".//figure"):
                    self._process_element(figure, state)
            return
        if tag == "table":
            summary_text, table_meta = _summarise_table(element)
            if summary_text and table_meta:
                state.add_text_block(
                    text=summary_text, kind="table_summary", table_meta=table_meta
                )
            return
        if tag == "figure":
            for img in element.xpath(".//img"):
                src = normalize_optional_string(img.get("src"))
                if not src:
                    continue
                alt_text = normalize_optional_string(img.get("alt"))
                caption_el = element.find(".//figcaption")
                caption_text = (
                    _collapse_whitespace("".join(caption_el.itertext()))
                    if caption_el is not None
                    else None
                )
                state.add_asset(
                    file_uri=src,
                    before_text=state._last_text,
                    after_text=caption_text,
                    alt_text=alt_text or caption_text,
                    declared_media_type=img.get("type"),
                )
            caption_el = element.find(".//figcaption")
            if caption_el is not None:
                caption_text = _render_block(
                    caption_el, state, allow_inline_assets=False
                )
                state.add_text_block(text=caption_text, kind="paragraph")
            return
        if tag == "img":
            src = normalize_optional_string(element.get("src"))
            if src:
                alt_text = normalize_optional_string(element.get("alt"))
                state.add_asset(
                    file_uri=src,
                    before_text=state._last_text,
                    after_text=None,
                    alt_text=alt_text,
                    declared_media_type=element.get("type"),
                )
            return
        if tag == "pre":
            code_text = "".join(element.itertext())
            state.add_text_block(text=code_text, kind="code")
            return
        leading_text = normalize_optional_string(element.text)
        if leading_text:
            state.add_text_block(text=leading_text, kind="paragraph")
        for child in element:
            self._process_element(child, state)
        trailing_text = normalize_optional_string(element.tail)
        if trailing_text:
            state.add_text_block(text=trailing_text, kind="paragraph")
