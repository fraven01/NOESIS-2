import base64
import hashlib
import io

import fitz  # type: ignore[import-untyped]
import pytest

from documents.contracts import InlineBlob
from documents.parsers_pdf import PdfDocumentParser
from documents.pipeline import DocumentPipelineConfig


_ONE_PIXEL_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


class _DocumentStub:
    def __init__(self, media_type: str, blob: InlineBlob) -> None:
        self.media_type = media_type
        self.blob = blob


def _inline_blob(payload: bytes) -> InlineBlob:
    return InlineBlob(
        type="inline",
        media_type="application/pdf",
        base64=base64.b64encode(payload).decode("ascii"),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def _create_pdf_with_content() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 96), "Project Overview", fontname="helv", fontsize=24)
    page.insert_text(
        (72, 140),
        "This is a paragraph describing the project.",
        fontname="helv",
        fontsize=12,
    )
    table_text = "Title  Count\nAlpha  2\nBeta  3"
    page.insert_text((72, 180), table_text, fontname="cour", fontsize=12)
    image_rect = fitz.Rect(200, 260, 240, 300)
    page.insert_image(image_rect, stream=_ONE_PIXEL_PNG)
    page.insert_text(
        (72, 320),
        "Image follow-up context.",
        fontname="helv",
        fontsize=12,
    )
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()
    return buffer.getvalue()


def _create_empty_pdf() -> bytes:
    document = fitz.open()
    document.new_page()
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()
    return buffer.getvalue()


def _create_encrypted_pdf() -> bytes:
    document = fitz.open()
    document.new_page()
    buffer = io.BytesIO()
    document.save(
        buffer,
        encryption=fitz.PDF_ENCRYPT_AES_256,  # type: ignore[attr-defined]
        owner_pw="owner",
        user_pw="secret",
    )
    document.close()
    return buffer.getvalue()


def test_pdf_parser_extracts_blocks_assets_and_tables() -> None:
    payload = _create_pdf_with_content()
    document = _DocumentStub("application/pdf", _inline_blob(payload))

    parser = PdfDocumentParser()
    result = parser.parse(document, DocumentPipelineConfig())

    texts_by_kind = {}
    for block in result.text_blocks:
        texts_by_kind.setdefault(block.kind, []).append(block)

    headings = texts_by_kind.get("heading", [])
    assert any(block.text == "Project Overview" for block in headings)

    tables = texts_by_kind.get("table_summary", [])
    assert tables
    table = tables[0]
    assert "Table" in table.text
    assert dict(table.table_meta or {})["headers"] == ["Title", "Count"]

    paragraphs = texts_by_kind.get("paragraph", [])
    assert any("paragraph describing" in block.text for block in paragraphs)

    assets = result.assets
    assert len(assets) == 1
    asset = assets[0]
    assert asset.media_type == "image/png"
    assert asset.context_before is not None and "Table" in asset.context_before
    assert asset.context_after == "Image follow-up context."
    assert asset.page_index == 0
    assert asset.bbox is not None and all(0.0 <= coord <= 1.0 for coord in asset.bbox)

    stats = result.statistics
    assert stats["parser.pages"] == 1
    assert stats["parse.blocks.total"] == len(result.text_blocks)
    assert stats["parse.assets.total"] == len(result.assets)
    assert stats["assets.images"] == 1
    assert stats["parser.tables"] >= 1
    assert {block.language for block in result.text_blocks} == {"en"}


def test_pdf_parser_triggers_ocr_for_empty_pages() -> None:
    payload = _create_empty_pdf()
    document = _DocumentStub("application/pdf", _inline_blob(payload))

    parser = PdfDocumentParser()
    result = parser.parse(document, DocumentPipelineConfig(enable_ocr=True))

    assert result.statistics.get("ocr.triggered_pages") == [0]


def test_pdf_parser_records_ocr_failure_codes() -> None:
    payload = _create_empty_pdf()
    document = _DocumentStub("application/pdf", _inline_blob(payload))

    def _renderer(**_: object) -> None:
        raise RuntimeError("boom")

    config = DocumentPipelineConfig(enable_ocr=True, ocr_renderer=_renderer)
    parser = PdfDocumentParser()
    result = parser.parse(document, config)

    assert result.statistics.get("ocr.triggered_pages") == [0]
    assert result.statistics.get("ocr.errors") == ["pdf_ocr_failed_page_0"]


def test_pdf_parser_raises_specific_error_for_encrypted_payload() -> None:
    payload = _create_encrypted_pdf()
    document = _DocumentStub("application/pdf", _inline_blob(payload))

    parser = PdfDocumentParser()

    with pytest.raises(ValueError, match="pdf_encrypted"):
        parser.parse(document, DocumentPipelineConfig())


def test_pdf_parser_language_detection_skips_short_text() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 96), "Hi")
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()

    payload = buffer.getvalue()
    result = PdfDocumentParser().parse(
        _DocumentStub("application/pdf", _inline_blob(payload)),
        DocumentPipelineConfig(),
    )

    assert all(block.language is None for block in result.text_blocks)
