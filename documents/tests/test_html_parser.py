from __future__ import annotations

import base64
import hashlib

from typing import Optional

import pytest

from documents.contracts import InlineBlob
from documents.parsers_html import HtmlDocumentParser
from documents.pipeline import DocumentPipelineConfig


def _inline_blob_from_bytes(
    payload: bytes, media_type: str = "text/html"
) -> InlineBlob:
    return InlineBlob(
        type="inline",
        media_type=media_type,
        base64=base64.b64encode(payload).decode("ascii"),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def _inline_blob_from_text(text: str, media_type: str = "text/html") -> InlineBlob:
    return _inline_blob_from_bytes(text.encode("utf-8"), media_type=media_type)


class _DocumentStub:
    def __init__(
        self,
        *,
        media_type: str,
        blob: InlineBlob,
        origin_uri: Optional[str] = None,
        language: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.media_type = media_type
        self.blob = blob
        self.file_name = name

        class _Meta:
            pass

        meta = _Meta()
        meta.origin_uri = origin_uri
        meta.language = language
        self.meta = meta


def test_html_parser_extracts_reader_content() -> None:
    document_html = """<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Sample Article</title>
    <meta charset="utf-8" />
  </head>
  <body>
    <header><nav>Navigation</nav></header>
    <main>
      <article>
        <h1>Title of Article</h1>
        <p>Intro paragraph with some leading text before the chart image <img src="images/chart.png" alt="Chart showing growth" /> continues after image.</p>
        <section>
          <h2>Details Section</h2>
          <p>First detail paragraph providing context.</p>
          <ul>
            <li>First bullet item.</li>
            <li>Second bullet item.</li>
          </ul>
          <table>
            <thead>
              <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
              <tr><td>Alpha</td><td>10</td></tr>
              <tr><td>Beta</td><td>20</td></tr>
              <tr><td>Gamma</td><td>30</td></tr>
            </tbody>
          </table>
          <figure>
            <img src="https://cdn.example.com/images/remote.jpg" alt="Remote" />
            <figcaption>Remote image caption.</figcaption>
          </figure>
        </section>
      </article>
    </main>
    <footer>Footer text</footer>
  </body>
</html>
"""
    document = _DocumentStub(
        media_type="text/html",
        blob=_inline_blob_from_text(document_html),
        origin_uri="https://example.com/articles/sample.html",
    )
    parser = HtmlDocumentParser()

    result = parser.parse(document, config={})

    headings = [block for block in result.text_blocks if block.kind == "heading"]
    paragraphs = [block for block in result.text_blocks if block.kind == "paragraph"]
    list_blocks = [block for block in result.text_blocks if block.kind == "list"]
    tables = [block for block in result.text_blocks if block.kind == "table_summary"]

    assert headings[0].text == "Title of Article"
    assert headings[0].section_path == ("Title of Article",)
    assert headings[1].text == "Details Section"
    assert headings[1].section_path == ("Title of Article", "Details Section")

    assert any(block.text.startswith("Intro paragraph") for block in paragraphs)
    assert any(block.text == "Remote image caption." for block in paragraphs)
    assert not any("Navigation" in block.text for block in result.text_blocks)

    assert list_blocks and "First bullet item." in list_blocks[0].text

    assert tables, "expected table summary block"
    table_meta = tables[0].table_meta or {}
    assert table_meta["rows"] == 3
    assert table_meta["columns"] == 2
    assert table_meta["headers"] == ["Metric", "Value"]
    assert table_meta["sample_row_count"] == 3
    assert table_meta["sample_rows"] == [
        ["Alpha", "10"],
        ["Beta", "20"],
        ["Gamma", "30"],
    ]
    assert len(tables[0].text) <= 500

    assert result.statistics["assets.images"] == 2
    assert result.statistics["parser.headings"] == 2
    assert result.statistics["parser.lists"] == 1
    assert result.statistics["parser.tables"] == 1
    assert result.statistics["parser.words"] > 0

    assets = result.assets
    assert {asset.file_uri for asset in assets} == {
        "images/chart.png",
        "https://cdn.example.com/images/remote.jpg",
    }
    for asset in assets:
        assert asset.media_type in {"image/png", "image/jpeg", "image/*"}
        if asset.context_before:
            assert len(asset.context_before) <= 512
        if asset.context_after:
            assert len(asset.context_after) <= 512
            assert (
                "Source: https://example.com/articles/sample.html"
                in asset.context_after
            )
        assert asset.metadata.get("parent_ref")
        assert asset.metadata.get("locator")
        assert (
            asset.metadata.get("origin_uri")
            == "https://example.com/articles/sample.html"
        )
        candidates = asset.metadata.get("caption_candidates", [])
        if asset.file_uri == "images/chart.png":
            assert candidates and candidates[0] == ("alt_text", "Chart showing growth")
        if asset.file_uri == "https://cdn.example.com/images/remote.jpg":
            assert candidates and candidates[0] == ("alt_text", "Remote")

    assert (
        result.statistics["assets.origins"]["images/chart.png"]
        == "https://example.com/articles/sample.html"
    )


def test_html_parser_can_handle_variants() -> None:
    parser = HtmlDocumentParser()

    html_doc = _DocumentStub(
        media_type="text/html",
        blob=_inline_blob_from_text("<html><body>Hi</body></html>"),
    )
    assert parser.can_handle(html_doc)

    ext_doc = _DocumentStub(
        media_type="text/plain",
        blob=_inline_blob_from_text(
            "<!DOCTYPE html><html><body>Hi</body></html>", media_type="text/plain"
        ),
        name="report.htm",
    )
    assert parser.can_handle(ext_doc)

    plain_doc = _DocumentStub(
        media_type="text/plain",
        blob=_inline_blob_from_text("Just some plain text", media_type="text/plain"),
    )
    assert not parser.can_handle(plain_doc)


def test_html_parser_handles_brotli_payload() -> None:
    brotli = pytest.importorskip("brotli")
    html_source = "<html><body><p>Brotli payload decoded.</p></body></html>"
    compressed = brotli.compress(html_source.encode("utf-8"))
    document = _DocumentStub(
        media_type="text/html",
        blob=_inline_blob_from_bytes(compressed),
        origin_uri="https://example.com/brotli",
    )
    parser = HtmlDocumentParser()

    result = parser.parse(document, config={})

    assert any(
        block.kind == "paragraph" and "Brotli payload decoded." in block.text
        for block in result.text_blocks
    )


def test_html_parser_readability_mode_reduces_boilerplate() -> None:
    document_html = """<!DOCTYPE html>
<html lang=\"en\">
  <head><title>News</title></head>
  <body>
    <div class=\"promo-banner\">Limited time offer! Subscribe now.</div>
    <div id=\"signup-callout\">Sign up today for exclusive updates.</div>
    <article>
      <h1>Breaking Story</h1>
      <p>Short summary of the situation with only a few words.</p>
    </article>
    <div class=\"related-links\">Related content and more promos.</div>
  </body>
</html>
"""
    document = _DocumentStub(
        media_type="text/html",
        blob=_inline_blob_from_text(document_html),
    )
    parser = HtmlDocumentParser()

    default_result = parser.parse(document, config={})
    readability_config = DocumentPipelineConfig(use_readability_html_extraction=True)
    readability_result = parser.parse(document, config=readability_config)

    assert any(
        "Limited time offer" in block.text for block in default_result.text_blocks
    )
    assert not any(
        "Limited time offer" in block.text for block in readability_result.text_blocks
    )
    assert any(
        block.kind == "heading" and block.text == "Breaking Story"
        for block in readability_result.text_blocks
    )
    assert any(
        block.kind == "paragraph" and "Short summary" in block.text
        for block in readability_result.text_blocks
    )


def test_html_parser_readability_fallback_when_summary_empty(monkeypatch) -> None:
    html_body = """
    <html>
      <body>
        <article><p>Main content retained.</p></article>
      </body>
    </html>
    """
    document = _DocumentStub(
        media_type="text/html",
        blob=_inline_blob_from_text(html_body),
    )
    parser = HtmlDocumentParser()

    monkeypatch.setattr(
        parser,
        "_extract_main_content_with_readability",
        lambda raw: None,
    )

    config = DocumentPipelineConfig(use_readability_html_extraction=True)
    result = parser.parse(document, config=config)

    assert any(block.text == "Main content retained." for block in result.text_blocks)
