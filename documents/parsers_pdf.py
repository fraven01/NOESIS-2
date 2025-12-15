"""PDF parser backed by PyMuPDF, pdfplumber and pikepdf."""

from __future__ import annotations

import io
import re
from contextlib import ExitStack, closing
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import fitz  # type: ignore[import-untyped]
import pdfplumber  # type: ignore[import-untyped]
import pikepdf  # type: ignore[import-untyped]
from fitz import Page as _FitzPage

from documents.contract_utils import (
    is_bcp47_like,
    normalize_media_type,
    normalize_string,
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

try:  # pragma: no cover - optional dependency exercised in integration environments
    from langdetect import DetectorFactory, LangDetectException, detect_langs  # type: ignore

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except Exception:  # pragma: no cover - deterministic fallback for minimal environments
    detect_langs = None  # type: ignore[assignment]
    LangDetectException = Exception  # type: ignore[assignment]
    _LANGDETECT_AVAILABLE = False

_PDF_MEDIA_TYPES = {"application/pdf"}
_PDF_MAX_ASSET_SIZE = 25 * 1024 * 1024
_PDF_MAX_TOTAL_ASSET_SIZE = 200 * 1024 * 1024
_HEADING_FONT_THRESHOLD = (
    18.0  # chosen to catch large headings while ignoring body text fonts
)
_TABLE_SAMPLE_LIMIT = 5


@dataclass
class _TableCandidate:
    bbox: Tuple[float, float, float, float]
    summary_text: str
    summary_meta: Mapping[str, Any]
    consumed: bool = False


class PdfDocumentParser(DocumentParser):
    """Parse PDF payloads using high level libraries with conservative heuristics."""

    def can_handle(self, document: Any) -> bool:
        # Check explicit media type (from metadata/staging)
        media_type = getattr(document, "media_type", None)
        if (
            isinstance(media_type, str)
            and normalize_media_type(media_type) in _PDF_MEDIA_TYPES
        ):
            return True

        # Check blob media type
        blob = getattr(document, "blob", None)
        blob_media = getattr(blob, "media_type", None)
        if (
            isinstance(blob_media, str)
            and normalize_media_type(blob_media) in _PDF_MEDIA_TYPES
        ):
            return True

        # Check meta.external_ref.media_type fallback
        # This handles cases where blob.media_type is None but the type is stored in metadata
        meta = getattr(document, "meta", None)
        external_ref = getattr(meta, "external_ref", None)
        if isinstance(external_ref, Mapping):
            ext_media = external_ref.get("media_type")
            if (
                isinstance(ext_media, str)
                and normalize_media_type(ext_media) in _PDF_MEDIA_TYPES
            ):
                return True

        # Fallback: Sniff payload (safe for Inline/Local blobs)
        try:
            payload = document_payload_bytes(document)
            return bool(payload and payload.startswith(b"%PDF"))
        except (ValueError, AttributeError):
            # Cannot access payload (e.g. storage missing for FileBlob), assume not PDF
            return False

    def parse(self, document: Any, config: Any) -> ParsedResult:
        try:
            payload = document_payload_bytes(document)
        except ValueError as exc:
            # Map storage/type errors to standard ValueError for consistency
            raise ValueError("pdf_blob_missing") from exc

        if not payload:
            raise ValueError("pdf_payload_missing")

        pdf_safe_mode = _get_config_flag(config, "pdf_safe_mode", True)
        enable_ocr = _get_config_flag(config, "enable_ocr", False)

        repaired_payload = _repair_pdf(payload)
        try:
            with ExitStack() as stack:
                pdf_document = stack.enter_context(
                    closing(fitz.open(stream=repaired_payload, filetype="pdf"))
                )
                try:
                    plumber_raw = pdfplumber.open(io.BytesIO(repaired_payload))
                except Exception:
                    plumber_document = None
                else:
                    plumber_document = stack.enter_context(closing(plumber_raw))

                text_blocks: List[ParsedTextBlock] = []
                assets: List[ParsedAsset] = []
                words = 0
                tables = 0
                images = 0
                total_asset_bytes = 0
                ocr_pages: List[int] = []
                ocr_errors: List[str] = []
                page_count = pdf_document.page_count

                for page_index in range(page_count):
                    page = pdf_document.load_page(page_index)
                    plumber_page = (
                        plumber_document.pages[page_index]
                        if plumber_document is not None
                        and page_index < len(plumber_document.pages)
                        else None
                    )
                    page_blocks = _safe_page_blocks(page)
                    table_candidates = list(_extract_tables(plumber_page))
                    page_language = _detect_page_language(page)

                    if enable_ocr and _page_is_empty(page):
                        ocr_pages.append(page_index)
                        error = _trigger_ocr_hook(config, page, page_index)
                        if error is not None:
                            ocr_errors.append(error)

                    section_path: List[str] = []
                    pending_assets: List[int] = []
                    last_text: Optional[str] = None

                    for block in page_blocks:
                        block_type = block.get("type")
                        if block_type == 0:
                            block_text, max_font = _extract_block_text(block)
                            if not block_text:
                                continue
                            normalized = (
                                normalize_string(block_text)
                                if pdf_safe_mode
                                else block_text.strip()
                            )
                            if not normalized:
                                continue

                            candidate = _match_table_candidate(
                                block.get("bbox"), table_candidates
                            )
                            fallback_table = (
                                None
                                if candidate is not None
                                else _detect_text_table(block_text)
                            )
                            if candidate is not None:
                                tables += 1
                                text_block = build_parsed_text_block(
                                    text=candidate.summary_text,
                                    kind="table_summary",
                                    section_path=(
                                        tuple(section_path) if section_path else None
                                    ),
                                    page_index=page_index,
                                    table_meta=candidate.summary_meta,
                                    language=page_language,
                                )
                            elif fallback_table is not None:
                                tables += 1
                                text_block = build_parsed_text_block(
                                    text=fallback_table[0],
                                    kind="table_summary",
                                    section_path=(
                                        tuple(section_path) if section_path else None
                                    ),
                                    page_index=page_index,
                                    table_meta=fallback_table[1],
                                    language=page_language,
                                )
                            else:
                                kind = _classify_block_kind(normalized, max_font)
                                if kind == "heading":
                                    # Truncate to 128 chars to comply with section_path validation
                                    section_path = [normalized[:128]]
                                text_block = build_parsed_text_block(
                                    text=normalized,
                                    kind=kind,  # type: ignore[arg-type]
                                    section_path=(
                                        tuple(section_path) if section_path else None
                                    ),
                                    page_index=page_index,
                                    language=page_language,
                                )
                                if kind == "heading":
                                    # Truncate to 128 chars to comply with section_path validation
                                    section_path = [normalized[:128]]

                            text_blocks.append(text_block)
                            words += len(text_block.text.split())
                            last_text = text_block.text
                            for asset_index in pending_assets:
                                assets[asset_index] = build_parsed_asset_with_meta(
                                    media_type=assets[asset_index].media_type,
                                    content=assets[asset_index].content,
                                    file_uri=assets[asset_index].file_uri,
                                    page_index=assets[asset_index].page_index,
                                    bbox=assets[asset_index].bbox,
                                    context_before=assets[asset_index].context_before,
                                    context_after=text_block.text,
                                    metadata={
                                        "asset_kind": "embedded_image",
                                        "locator": f"document:image:{len(assets)+1}",
                                    },
                                )
                            pending_assets.clear()
                            continue

                        if block_type == 1:
                            asset = _extract_asset_from_block(
                                page,
                                block,
                                page_index,
                                last_text,
                            )
                            if asset is None:
                                continue
                            if (
                                asset.content is not None
                                and len(asset.content) > _PDF_MAX_ASSET_SIZE
                            ):
                                raise ValueError("pdf_asset_too_large")
                            asset_size = len(asset.content or b"")
                            total_asset_bytes += asset_size
                            if total_asset_bytes > _PDF_MAX_TOTAL_ASSET_SIZE:
                                raise ValueError("pdf_assets_total_too_large")
                            assets.append(asset)
                            pending_assets.append(len(assets) - 1)
                            images += 1

                    for candidate in table_candidates:
                        if candidate.consumed:
                            continue
                        tables += 1
                        text_blocks.append(
                            build_parsed_text_block(
                                text=candidate.summary_text,
                                kind="table_summary",
                                section_path=(
                                    tuple(section_path) if section_path else None
                                ),
                                page_index=page_index,
                                table_meta=candidate.summary_meta,
                                language=page_language,
                            )
                        )

                statistics: Dict[str, Any] = {
                    "parser.pages": page_count,
                    "parser.words": words,
                    "parser.tables": tables,
                    "assets.images": images,
                    "parse.blocks.total": len(text_blocks),
                    "parse.assets.total": len(assets),
                }
                if ocr_pages:
                    statistics["ocr.triggered_pages"] = ocr_pages
                if ocr_errors:
                    statistics["ocr.errors"] = ocr_errors

                return build_parsed_result(
                    text_blocks=tuple(text_blocks),
                    assets=tuple(assets),
                    statistics=statistics,
                )
        except fitz.fitz.FileDataError as exc:  # type: ignore[attr-defined]
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                raise ValueError("pdf_encrypted") from exc
            raise ValueError("pdf_open_failed") from exc


def _page_is_empty(page: _FitzPage) -> bool:
    text = page.get_text("text")
    return not text or not text.strip()


def _detect_page_language(page: _FitzPage) -> Optional[str]:
    text = page.get_text("text")
    return _detect_language_from_text(text)


def _detect_language_from_text(text: Optional[str]) -> Optional[str]:
    if not text or not _LANGDETECT_AVAILABLE or detect_langs is None:
        return None
    normalised = normalize_string(text)
    if len(normalised) < 20:
        return None
    sample = normalised[:2000]
    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        return None
    if not candidates:
        return None
    top = max(candidates, key=lambda entry: entry.prob)
    if top.prob < 0.8:
        return None
    candidate = top.lang
    if candidate and is_bcp47_like(candidate):
        return candidate
    return None


def _trigger_ocr_hook(config: Any, page: _FitzPage, page_index: int) -> Optional[str]:
    renderer: Optional[Callable[..., Any]] = getattr(config, "ocr_renderer", None)
    if renderer is None:
        return None
    if not callable(renderer):
        return f"pdf_ocr_failed_page_{page_index}"
    try:
        pixmap = page.get_pixmap()
        image_bytes = pixmap.tobytes("png")
    except Exception:
        return f"pdf_ocr_failed_page_{page_index}"
    try:
        renderer(page_index=page_index, image=image_bytes)
    except Exception:
        return f"pdf_ocr_failed_page_{page_index}"
    return None


def _safe_page_blocks(page: _FitzPage) -> Sequence[Mapping[str, Any]]:
    dictionary = page.get_text("dict")
    blocks = dictionary.get("blocks", [])
    return blocks if isinstance(blocks, list) else []


def _extract_block_text(block: Mapping[str, Any]) -> Tuple[str, float]:
    max_font = 0.0
    lines = block.get("lines", [])
    parts: List[str] = []
    for line in lines:
        spans = line.get("spans", [])
        span_texts: List[str] = []
        for span in spans:
            text = span.get("text", "")
            if not isinstance(text, str):
                continue
            span_texts.append(text)
            try:
                size = float(span.get("size", 0.0))
            except (TypeError, ValueError):
                size = 0.0
            if size > max_font:
                max_font = size
        if span_texts:
            parts.append("".join(span_texts))
    return ("\n".join(parts), max_font)


def _extract_asset_from_block(
    page: _FitzPage,
    block: Mapping[str, Any],
    page_index: int,
    context_before: Optional[str],
) -> Optional[ParsedAsset]:
    image = block.get("image")
    content: Optional[bytes] = None
    media_type = "image/unknown"
    if isinstance(image, Mapping):
        xref = image.get("xref")
        if not isinstance(xref, int):
            return None
        try:
            extracted = page.parent.extract_image(xref)
        except ValueError:
            return None
        raw = extracted.get("image")
        if not isinstance(raw, (bytes, bytearray)):
            return None
        content = bytes(raw)
        ext = extracted.get("ext")
        if isinstance(ext, str) and ext:
            media_type = normalize_media_type(f"image/{ext.lower()}")
    elif isinstance(image, (bytes, bytearray)):
        content = bytes(image)
        ext = block.get("ext")
        if isinstance(ext, str) and ext:
            media_type = normalize_media_type(f"image/{ext.lower()}")
    else:
        return None

    if content is None:
        return None

    bbox = block.get("bbox")
    normalized_bbox = _normalise_bbox(bbox, page.rect)
    return build_parsed_asset_with_meta(
        media_type=media_type,
        content=content,
        page_index=page_index,
        bbox=normalized_bbox,
        context_before=context_before,
        metadata={"asset_kind": "embedded_image", "locator": "page_image"},
    )


def _normalise_bbox(
    bbox: Optional[Sequence[float]],
    rect: fitz.Rect,  # type: ignore[valid-type]
) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    width = rect.width or 1.0
    height = rect.height or 1.0
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, min(1.0, float(x0) / width)),
        max(0.0, min(1.0, float(y0) / height)),
        max(0.0, min(1.0, float(x1) / width)),
        max(0.0, min(1.0, float(y1) / height)),
    )


def _classify_block_kind(text: str, max_font: float) -> str:
    stripped = text.lstrip()
    if _looks_like_list(stripped):
        return "list"
    if max_font >= _HEADING_FONT_THRESHOLD:
        return "heading"
    return "paragraph"


def _looks_like_list(text: str) -> bool:
    if not text:
        return False
    prefixes = ("•", "-", "*", "–")
    if text[0] in prefixes:
        return True
    if text[:2].isdigit():
        return True
    if text[:3].strip().rstrip(".").isdigit():
        return True
    return False


def _extract_tables(
    plumber_page: Optional[pdfplumber.page.Page],
) -> Iterable[_TableCandidate]:
    if plumber_page is None:
        return []
    candidates: List[_TableCandidate] = []
    try:
        tables = plumber_page.find_tables()
    except Exception:
        tables = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        summary = _summarize_table(rows)
        if summary is None:
            continue
        text, meta = summary
        candidates.append(
            _TableCandidate(
                bbox=table.bbox,
                summary_text=text,
                summary_meta=meta,
            )
        )
    return candidates


def _match_table_candidate(
    bbox: Optional[Sequence[float]],
    candidates: Sequence[_TableCandidate],
) -> Optional[_TableCandidate]:
    if not bbox or len(bbox) != 4:
        return None
    for candidate in candidates:
        if candidate.consumed:
            continue
        if _bbox_intersects(bbox, candidate.bbox):
            candidate.consumed = True
            return candidate
    return None


def _bbox_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _detect_text_table(text: str) -> Optional[Tuple[str, Mapping[str, Any]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    delimiter = re.compile(r"\t+|\s{2,}")
    rows: List[List[str]] = []
    expected_width: Optional[int] = None
    for line in lines:
        parts = [
            normalize_string(part) for part in delimiter.split(line) if part.strip()
        ]
        if len(parts) < 2:
            return None
        if expected_width is None:
            expected_width = len(parts)
        elif len(parts) != expected_width:
            return None
        rows.append(parts)
    if len(rows) < 2:
        return None
    return _summarize_table(rows)


def _summarize_table(
    rows: Optional[Sequence[Sequence[str]]],
) -> Optional[Tuple[str, Mapping[str, Any]]]:
    if not rows:
        return None
    cleaned: List[List[str]] = []
    for row in rows:
        if not isinstance(row, Sequence):
            continue
        cleaned.append([normalize_string(cell or "") for cell in row])
    if not cleaned:
        return None
    headers = cleaned[0]
    body = cleaned[1:]
    columns = max(len(row) for row in cleaned)
    rows_count = len(cleaned)
    sample_rows = [row[:columns] for row in body[:_TABLE_SAMPLE_LIMIT]]
    meta: Dict[str, Any] = {
        "rows": rows_count,
        "columns": columns,
        "headers": headers,
        "sample_row_count": len(sample_rows),
        "sample_rows": sample_rows,
    }
    summary_lines = [
        f"Table {rows_count}x{columns}",
        "Headers: " + ", ".join(headers),
    ]
    if sample_rows:
        preview = "; ".join([" | ".join(row) for row in sample_rows])
        summary_lines.append(f"Samples: {preview}")
    summary_text = " \u2014 ".join(summary_lines)
    if len(summary_text) > 500:
        summary_text = summary_text[:500]
    return summary_text, meta


def _repair_pdf(payload: bytes) -> bytes:
    try:
        with pikepdf.open(io.BytesIO(payload)) as pdf:
            buffer = io.BytesIO()
            pdf.save(buffer)
            return buffer.getvalue()
    except pikepdf.PasswordError as exc:  # type: ignore[attr-defined]
        raise ValueError("pdf_encrypted") from exc
    except pikepdf.PdfError:
        return payload


def _get_config_flag(config: Any, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    return bool(value)
