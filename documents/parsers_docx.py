"""DOCX parser implementation producing structured parser results."""

from __future__ import annotations

import io
import posixpath
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from docx import Document as _load_docx_document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from documents.contract_utils import (
    is_bcp47_like,
    normalize_media_type,
    normalize_string,
    truncate_text,
)
from documents.normalization import document_payload_bytes
from documents.parsers import (
    DocumentParser,
    ParsedAsset,
    ParsedResult,
    ParsedTextBlock,
    build_parsed_asset_with_meta,
    build_parsed_result,
    build_parsed_text_block,
)


_DOCX_MEDIA_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_DOCX_MAX_ARCHIVE_ENTRIES = 10_000
_DOCX_MAX_UNCOMPRESSED_SIZE = 500 * 1024 * 1024
_DOCX_MAX_ASSET_SIZE = 25 * 1024 * 1024
_TABLE_PREVIEW_LIMIT = 500

_IMAGE_EXTENSION_FALLBACK = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def _iter_block_items(document: _Document) -> Iterable[Tuple[str, Any]]:
    """Yield block-level elements from a python-docx document in order."""

    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield ("paragraph", Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            yield ("table", Table(child, document))


@dataclass
class _AssetBuilder:
    media_type: str
    content: bytes
    context_before: Optional[str]
    context_after: Optional[str] = None
    alt_text: Optional[str] = None
    parent_ref: Optional[str] = None
    locator: Optional[str] = None


def _normalise_media_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return normalize_media_type(value)
    except ValueError:
        return None


def _looks_like_docx(payload: Optional[bytes]) -> bool:
    if not payload:
        return False
    if not payload.startswith(b"PK"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            try:
                _validate_archive(archive)
            except ValueError:
                return False
            names = {name for name in archive.namelist()}
            return "word/document.xml" in names
    except zipfile.BadZipFile:
        return False


def _parse_content_types(archive: zipfile.ZipFile) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        with archive.open("[Content_Types].xml") as handle:
            root = ET.parse(handle).getroot()
    except KeyError:
        return mapping
    namespace = "http://schemas.openxmlformats.org/package/2006/content-types"
    default_tag = f"{{{namespace}}}Default"
    override_tag = f"{{{namespace}}}Override"
    for child in root:
        if child.tag == default_tag:
            extension = child.get("Extension")
            content_type = child.get("ContentType")
            if extension and content_type:
                mapping[f".{extension.lower()}"] = content_type
        elif child.tag == override_tag:
            part = child.get("PartName")
            content_type = child.get("ContentType")
            if part and content_type:
                mapping[part.lstrip("/")] = content_type
    return mapping


def _load_app_properties(archive: zipfile.ZipFile) -> Mapping[str, int]:
    try:
        with archive.open("docProps/app.xml") as handle:
            root = ET.parse(handle).getroot()
    except KeyError:
        return {}
    properties: Dict[str, int] = {}
    namespace = (
        "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    )
    for child in root:
        if child.tag.startswith(f"{{{namespace}}}") and child.text:
            name = child.tag.split("}", 1)[1]
            try:
                properties[name.lower()] = int(child.text.strip())
            except ValueError:
                continue
    return properties


def _load_core_properties(archive: zipfile.ZipFile) -> Mapping[str, str]:
    try:
        with archive.open("docProps/core.xml") as handle:
            root = ET.parse(handle).getroot()
    except KeyError:
        return {}
    properties: Dict[str, str] = {}
    namespace = {
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    language = root.find("dc:language", namespaces=namespace)
    if language is not None and language.text:
        candidate = normalize_string(language.text)
        if candidate:
            properties["language"] = candidate
    return properties


def _load_styles_language(archive: zipfile.ZipFile) -> Optional[str]:
    try:
        with archive.open("word/styles.xml") as handle:
            root = ET.parse(handle).getroot()
    except KeyError:
        return None
    for element in root.findall(".//w:lang", namespaces=_NS):
        for attribute in ("val", "eastAsia", "bidi"):
            candidate = element.get(f"{{{_NS['w']}}}{attribute}")
            if not candidate:
                continue
            normalised = normalize_string(candidate)
            if normalised:
                return normalised
    return None


def _validate_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > _DOCX_MAX_ARCHIVE_ENTRIES:
        raise ValueError("docx_archive_limits")
    total_size = 0
    for info in infos:
        total_size += max(info.file_size, 0)
        if total_size > _DOCX_MAX_UNCOMPRESSED_SIZE:
            raise ValueError("docx_archive_limits")


def _iter_paragraph_items(element: ET.Element) -> Iterable[Tuple[str, Any]]:
    for child in element:
        if child.tag == f"{{{_NS['w']}}}r":
            yield from _iter_run_items(child)
        elif child.tag in {
            f"{{{_NS['w']}}}hyperlink",
            f"{{{_NS['w']}}}smartTag",
            f"{{{_NS['w']}}}sdt",
            f"{{{_NS['w']}}}fldSimple",
        }:
            yield from _iter_paragraph_items(child)


def _iter_run_items(run: ET.Element) -> Iterable[Tuple[str, Any]]:
    texts: List[str] = []
    for child in run:
        if child.tag == f"{{{_NS['w']}}}t":
            value = child.text or ""
            if child.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve":
                texts.append(value)
            else:
                texts.append(value.strip())
        elif child.tag == f"{{{_NS['w']}}}tab":
            texts.append("\t")
        elif child.tag == f"{{{_NS['w']}}}br":
            texts.append(" ")
        elif child.tag == f"{{{_NS['w']}}}drawing":
            if texts:
                yield ("text", "".join(texts))
                texts.clear()
            alt_text = _extract_alt_text(child)
            for blip in child.findall(".//a:blip", namespaces=_NS):
                embed = blip.get(f"{{{_NS['r']}}}embed")
                if embed:
                    yield ("image", {"embed": embed, "alt": alt_text})
        else:
            yield from _iter_run_items(child)
    if texts:
        yield ("text", "".join(texts))


def _normalise_fragment(value: str) -> str:
    if not value:
        return ""
    collapsed = value.replace("\n", " ")
    return normalize_string(collapsed)


def _heading_level(paragraph: ET.Element) -> Optional[int]:
    ppr = paragraph.find("w:pPr", namespaces=_NS)
    if ppr is None:
        return None
    outline = ppr.find("w:outlineLvl", namespaces=_NS)
    if outline is not None:
        val = outline.get(f"{{{_NS['w']}}}val")
        if val is not None:
            try:
                return int(val) + 1
            except ValueError:
                pass
    style = ppr.find("w:pStyle", namespaces=_NS)
    if style is not None:
        raw = style.get(f"{{{_NS['w']}}}val") or ""
        candidate = normalize_string(raw).lower()
        if candidate.startswith("heading"):
            suffix = candidate[len("heading") :]
            if suffix.isdigit():
                level = int(suffix)
                if 1 <= level <= 6:
                    return level
    return None


def _is_list_paragraph(paragraph: ET.Element) -> bool:
    ppr = paragraph.find("w:pPr", namespaces=_NS)
    if ppr is None:
        return False
    if ppr.find("w:numPr", namespaces=_NS) is not None:
        return True
    style = ppr.find("w:pStyle", namespaces=_NS)
    if style is not None:
        raw = style.get(f"{{{_NS['w']}}}val") or ""
        candidate = normalize_string(raw).lower()
        if "list" in candidate or "bullet" in candidate:
            return True
    return False


def _collect_table_cells(table: Table) -> List[List[str]]:
    rows: List[List[str]] = []
    for row in table.rows:
        cells: List[str] = []
        for cell in row.cells:
            fragments: List[str] = []
            for paragraph in cell.paragraphs:
                texts = [
                    _normalise_fragment(value)
                    for kind, value in _iter_paragraph_items(paragraph._element)
                    if kind == "text"
                ]
                paragraph_text = " ".join(fragment for fragment in texts if fragment)
                if paragraph_text:
                    fragments.append(paragraph_text)
            cell_text = " ".join(fragment for fragment in fragments if fragment)
            cells.append(cell_text)
        rows.append(cells)
    return rows


def _table_summary(table: Table) -> Tuple[Optional[str], Mapping[str, Any]]:
    rows = _collect_table_cells(table)
    if not rows:
        return None, {}
    data_rows = rows[1:] if len(rows) > 1 else []
    row_count = len(data_rows)
    column_count = max((len(row) for row in rows), default=0)
    header: Sequence[str] = ()
    if rows:
        first = rows[0]
        header = tuple(first)
    samples = [
        "\t".join(filter(None, row)) for row in data_rows if any(cell for cell in row)
    ]
    sample_rows = [list(row) for row in data_rows[:5] if any(cell for cell in row)]
    sample_text = "\n".join(samples[:5])
    summary_parts = [f"rows={row_count}", f"columns={column_count}"]
    header_names = ", ".join(filter(None, header))
    if header_names:
        summary_parts.append(f"headers={header_names}")
    if sample_text:
        summary_parts.append("sample:\n" + sample_text)
    summary_text = "; ".join(summary_parts)
    summary_text = truncate_text(summary_text, _TABLE_PREVIEW_LIMIT) or ""
    summary_text = normalize_string(summary_text)
    if not summary_text:
        return None, {"rows": row_count, "columns": column_count}
    table_meta: Dict[str, Any] = {
        "rows": row_count,
        "columns": column_count,
    }
    if header_names:
        table_meta["headers"] = list(filter(None, header))
    if sample_rows:
        table_meta["sample_row_count"] = len(sample_rows)
        table_meta["sample_rows"] = sample_rows
    return summary_text, table_meta


def _extract_alt_text(drawing: ET.Element) -> Optional[str]:
    doc_pr = drawing.find(".//wp:docPr", namespaces=_NS)
    if doc_pr is not None:
        for attribute in ("descr", "title"):
            candidate = doc_pr.get(attribute)
            if candidate:
                normalised = normalize_string(candidate)
                if normalised:
                    return normalised
    # Fallback to picture-specific metadata when docPr is absent.
    c_nv_pr = drawing.find(".//pic:cNvPr", namespaces=_NS)
    if c_nv_pr is not None:
        for attribute in ("descr", "name"):
            candidate = c_nv_pr.get(attribute)
            if candidate:
                normalised = normalize_string(candidate)
                if normalised:
                    return normalised
    return None


class DocxDocumentParser(DocumentParser):
    """Parse DOCX payloads into structured text blocks and assets."""

    def can_handle(self, document: Any) -> bool:
        media_type = getattr(document, "media_type", None)
        if (
            isinstance(media_type, str)
            and normalize_media_type(media_type) in _DOCX_MEDIA_TYPES
        ):
            return True

        blob = getattr(document, "blob", None)
        blob_media = getattr(blob, "media_type", None)
        if (
            isinstance(blob_media, str)
            and normalize_media_type(blob_media) in _DOCX_MEDIA_TYPES
        ):
            return True

        # Fallback: looks like docx check
        try:
            payload = document_payload_bytes(document)
            return _looks_like_docx(payload)
        except (ValueError, AttributeError):
            return False

    def parse(self, document: Any, config: Any) -> ParsedResult:
        try:
            payload = document_payload_bytes(document)
        except ValueError as exc:
            raise ValueError("docx_blob_missing") from exc

        if not payload:
            raise ValueError("docx_blob_missing")
        try:
            archive_stream = io.BytesIO(payload)
            with zipfile.ZipFile(archive_stream) as archive:
                _validate_archive(archive)
                content_types = _parse_content_types(archive)
                app_props = _load_app_properties(archive)
                core_props = _load_core_properties(archive)
                styles_language = _load_styles_language(archive)
        except zipfile.BadZipFile as exc:  # pragma: no cover - defensive guard
            raise ValueError("docx_invalid_package") from exc

        archive_stream.seek(0)
        try:
            document_obj = _load_docx_document(archive_stream)
        except KeyError as exc:  # pragma: no cover - dependent on python-docx internals
            raise ValueError("docx_image_missing") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError("docx_invalid_package") from exc

        assets: List[_AssetBuilder] = []
        pending_after: List[_AssetBuilder] = []
        text_blocks: List[ParsedTextBlock] = []
        section_stack: List[str] = []
        last_text_block: Optional[str] = None
        tables_count = 0
        images_count = 0
        word_count = 0
        document_language: Optional[str] = None
        paragraph_counter = 0
        heading_counts: Dict[int, int] = {}
        global_image_index = 0

        for candidate in (
            core_props.get("language"),
            styles_language,
        ):
            if candidate and is_bcp47_like(candidate):
                document_language = candidate
                break

        related_parts = document_obj.part.related_parts

        def fulfill_pending(text: Optional[str]) -> None:
            nonlocal pending_after
            if not text:
                return
            for builder in pending_after:
                if builder.context_after is None:
                    builder.context_after = text
            pending_after = [
                builder for builder in pending_after if builder.context_after is None
            ]

        for block_type, block in _iter_block_items(document_obj):
            if block_type == "paragraph":
                element = block._element
                items = list(_iter_paragraph_items(element))
                if not items and not element.findall(".//w:drawing", namespaces=_NS):
                    continue
                normalized_items: List[Tuple[str, Any]] = []
                for kind, value in items:
                    if kind == "text":
                        fragment = _normalise_fragment(value)
                        if fragment:
                            normalized_items.append(("text", fragment))
                    else:
                        normalized_items.append((kind, value))

                level = _heading_level(element)
                is_list = _is_list_paragraph(element)
                paragraph_counter += 1
                if level is not None:
                    heading_counts[level] = heading_counts.get(level, 0) + 1
                    current_parent_ref = f"heading:{level}:{heading_counts[level]}"
                elif is_list:
                    current_parent_ref = f"list:{paragraph_counter}"
                else:
                    current_parent_ref = f"paragraph:{paragraph_counter}"

                paragraph_fragments = [
                    value for kind, value in normalized_items if kind == "text"
                ]
                paragraph_text = " ".join(paragraph_fragments).strip()

                for index, (kind, value) in enumerate(normalized_items):
                    if kind != "image":
                        continue
                    embed = value.get("embed") if isinstance(value, Mapping) else None
                    if not embed:
                        continue
                    alt_text = value.get("alt") if isinstance(value, Mapping) else None
                    image_part = related_parts.get(embed)
                    if image_part is None:
                        raise ValueError("docx_image_missing")
                    content = getattr(image_part, "blob", None)
                    if content is None:
                        raise ValueError("docx_image_missing")
                    if len(content) > _DOCX_MAX_ASSET_SIZE:
                        raise ValueError("docx_asset_too_large")
                    content_type = _normalise_media_type(
                        getattr(image_part, "content_type", None)
                    )
                    partname = getattr(image_part, "partname", None)
                    extension = ""
                    part_str = None
                    if partname is not None:
                        part_str = str(partname)
                        extension = posixpath.splitext(part_str)[1].lower()
                    if not content_type or not content_type.startswith("image/"):
                        fallback: Optional[str] = None
                        if part_str is not None:
                            fallback = content_types.get(part_str.lstrip("/"))
                        if extension:
                            extension_type = content_types.get(extension)
                            if extension_type and extension_type.startswith("image/"):
                                fallback = extension_type
                            elif not extension_type:
                                fallback = _IMAGE_EXTENSION_FALLBACK.get(extension)
                            else:
                                alt = _IMAGE_EXTENSION_FALLBACK.get(extension)
                                if alt:
                                    fallback = alt
                        if fallback and fallback.startswith("image/"):
                            content_type = _normalise_media_type(fallback)
                    content_type = content_type or "image/unknown"
                    before_fragments = [
                        val
                        for kind2, val in normalized_items[:index]
                        if kind2 == "text"
                    ]
                    before_text = " ".join(before_fragments).strip()
                    if not before_text:
                        before_text = last_text_block or ""
                    after_fragments = [
                        val
                        for kind2, val in normalized_items[index + 1 :]
                        if kind2 == "text"
                    ]
                    after_text = " ".join(after_fragments).strip()
                    if not after_text and alt_text:
                        after_text = alt_text
                    global_image_index += 1
                    if embed:
                        locator = f"{current_parent_ref}:rid:{embed}"
                    else:
                        locator = f"{current_parent_ref}:image:{global_image_index}"
                    builder = _AssetBuilder(
                        media_type=content_type,
                        content=content,
                        context_before=before_text or None,
                        context_after=after_text or None,
                        alt_text=alt_text or None,
                        parent_ref=current_parent_ref,
                        locator=locator,
                    )
                    if builder.context_after is None:
                        pending_after.append(builder)
                    assets.append(builder)
                    images_count += 1

                if paragraph_text:
                    if level is not None:
                        while len(section_stack) >= level:
                            section_stack.pop()
                        # Truncate to 128 chars to comply with section_path validation
                        section_stack.append(paragraph_text[:128])
                        section_path: Optional[Tuple[str, ...]] = tuple(section_stack)
                        block_obj = build_parsed_text_block(
                            text=paragraph_text,
                            kind="heading",
                            section_path=section_path,
                            language=document_language,
                        )
                    else:
                        section_path = tuple(section_stack) if section_stack else None
                        kind = "list" if is_list else "paragraph"
                        block_obj = build_parsed_text_block(
                            text=paragraph_text,
                            kind=kind,
                            section_path=section_path,
                            language=document_language,
                        )
                    text_blocks.append(block_obj)
                    fulfill_pending(paragraph_text)
                    last_text_block = paragraph_text
                    word_count += len(paragraph_text.split())
            elif block_type == "table":
                summary_text, table_meta = _table_summary(block)
                if not summary_text:
                    continue
                section_path = tuple(section_stack) if section_stack else None
                block_obj = build_parsed_text_block(
                    text=summary_text,
                    kind="table_summary",
                    section_path=section_path,
                    table_meta=table_meta,
                    language=document_language,
                )
                text_blocks.append(block_obj)
                fulfill_pending(summary_text)
                last_text_block = summary_text
                tables_count += 1
                word_count += len(summary_text.split())

        for builder in pending_after:
            if not builder.context_after:
                builder.context_after = last_text_block

        parsed_assets: List[ParsedAsset] = []
        for builder in assets:
            metadata: Dict[str, Any] = {}
            candidates: List[Tuple[str, str]] = []
            if builder.alt_text:
                candidates.append(("alt_text", builder.alt_text))
            if builder.context_after:
                candidates.append(("context_after", builder.context_after))
            if builder.context_before:
                candidates.append(("context_before", builder.context_before))
            if candidates:
                metadata["caption_candidates"] = candidates
            if builder.parent_ref:
                metadata["parent_ref"] = builder.parent_ref
            if builder.locator:
                metadata["locator"] = builder.locator
            parsed_assets.append(
                build_parsed_asset_with_meta(
                    media_type=builder.media_type,
                    content=builder.content,
                    context_before=builder.context_before,
                    context_after=builder.context_after,
                    metadata=metadata,
                )
            )

        statistics = {
            "parser.pages": app_props.get("pages", 0),
            "parser.words": max(app_props.get("words", 0), word_count),
            "parser.tables": max(app_props.get("tables", tables_count), tables_count),
            "assets.images": max(app_props.get("images", images_count), images_count),
        }

        return build_parsed_result(
            text_blocks=tuple(text_blocks),
            assets=tuple(parsed_assets),
            statistics=statistics,
        )
