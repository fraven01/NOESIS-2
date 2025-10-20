from __future__ import annotations

import base64
import hashlib

import pytest

from typing import Optional

from documents.contracts import InlineBlob
from documents.parsers_markdown import MarkdownDocumentParser


def _inline_blob_from_text(text: str, media_type: str = "text/markdown") -> InlineBlob:
    payload = text.encode("utf-8")
    return InlineBlob(
        type="inline",
        media_type=media_type,
        base64=base64.b64encode(payload).decode("ascii"),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


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


def test_markdown_parser_handles_front_matter() -> None:
    markdown = """---
title: Sample Doc
tags:
  - Alpha
  - beta
language: en-US
author: Jane Doe
date: 2024-05-01
---
# Overview

First paragraph.
"""
    document = _DocumentStub(
        media_type="text/markdown", blob=_inline_blob_from_text(markdown)
    )
    parser = MarkdownDocumentParser()

    result = parser.parse(document, config={})

    assert result.statistics["parser.front_matter"] is True
    assert result.statistics["front_matter.title"] == "Sample Doc"
    assert result.statistics["front_matter.tags"] == ["alpha", "beta"]
    assert result.statistics["front_matter.language"] == "en-US"
    assert result.statistics["front_matter.author"] == "Jane Doe"
    assert result.statistics["front_matter.date"] == "2024-05-01"
    assert any(block.kind == "heading" and block.text == "Overview" for block in result.text_blocks)
    assert all("Sample Doc" not in block.text for block in result.text_blocks)


def test_markdown_parser_extracts_headings_lists_and_footnotes() -> None:
    markdown = """# Title

Intro paragraph with footnote[^1].

## Details

- item one
- [x] item two

[^1]: Footnote text
"""
    document = _DocumentStub(
        media_type="text/markdown", blob=_inline_blob_from_text(markdown)
    )
    parser = MarkdownDocumentParser()

    result = parser.parse(document, config={})

    heading_blocks = [block for block in result.text_blocks if block.kind == "heading"]
    list_blocks = [block for block in result.text_blocks if block.kind == "list"]
    paragraph_blocks = [block for block in result.text_blocks if block.kind == "paragraph"]

    assert heading_blocks[0].text == "Title"
    assert heading_blocks[0].section_path == ("Title",)
    assert heading_blocks[1].text == "Details"
    assert heading_blocks[1].section_path == ("Title", "Details")

    assert any(block.text.endswith("[fn: Footnote text]") for block in paragraph_blocks)

    assert any(block.text == "item one" for block in list_blocks)
    assert any(block.text.endswith("item two") for block in list_blocks)
    for block in list_blocks:
        assert block.section_path and block.section_path[-1] == "Details"


def test_markdown_parser_tables_and_code_blocks() -> None:
    markdown = """| Name | Count |
| ---- | ----- |
| Alpha | 2 |
| Beta | 3 |

```python
print("hello")
```

```mermaid
graph TD
```
"""
    document = _DocumentStub(
        media_type="text/markdown", blob=_inline_blob_from_text(markdown)
    )
    parser = MarkdownDocumentParser()

    result = parser.parse(document, config={})

    table_blocks = [block for block in result.text_blocks if block.kind == "table_summary"]
    code_blocks = [block for block in result.text_blocks if block.kind == "code"]

    assert table_blocks, "expected a table summary block"
    table_meta = table_blocks[0].table_meta or {}
    assert table_meta["rows"] == 2
    assert table_meta["columns"] == 2
    assert table_meta["headers"] == ["Name", "Count"]
    assert table_meta["sample_row_count"] == 2
    assert table_meta["sample_rows"] == [["Alpha", "2"], ["Beta", "3"]]
    assert len(table_blocks[0].text) <= 500

    assert len(code_blocks) == 2
    python_meta = code_blocks[0].table_meta or {}
    assert python_meta["lang"] == "python"
    mermaid_meta = code_blocks[1].table_meta or {}
    assert mermaid_meta["lang"] == "mermaid"


def test_markdown_parser_images_and_html_assets() -> None:
    markdown = """Intro paragraph before image.

![Diagram](images/diagram.png "Diagram")

More text after image.

![Remote](https://example.com/image.jpg)

<figure>
  <img src="media/photo.gif" alt="Figure photo" />
  <figcaption>Figure caption text.</figcaption>
</figure>
"""
    document = _DocumentStub(
        media_type="text/markdown",
        blob=_inline_blob_from_text(markdown),
        origin_uri="https://example.com/docs/sample.md",
        language="en-US",
    )
    parser = MarkdownDocumentParser()

    result = parser.parse(document, config={})

    assert result.statistics["assets.images"] == 3
    assets = result.assets
    assert len(assets) == 3
    file_uris = {asset.file_uri for asset in assets}
    assert "images/diagram.png" in file_uris
    assert "https://example.com/image.jpg" in file_uris
    assert "media/photo.gif" in file_uris

    for asset in assets:
        if asset.context_before:
            assert len(asset.context_before) <= 512
        if asset.context_after:
            assert len(asset.context_after) <= 512
            assert "Source: https://example.com/docs/sample.md" in asset.context_after

    origins = result.statistics["assets.origins"]
    assert all(value == "https://example.com/docs/sample.md" for value in origins.values())

    assert any(
        block.kind == "paragraph" and block.text == "Figure caption text."
        for block in result.text_blocks
    )


def test_markdown_parser_collapses_whitespace_runs() -> None:
    markdown = """# Title

Paragraph with    multiple\tspaces and
line breaks.

Another   paragraph after   code:

```
code sample
```

Final    line.
"""
    document = _DocumentStub(
        media_type="text/markdown", blob=_inline_blob_from_text(markdown)
    )
    parser = MarkdownDocumentParser()

    result = parser.parse(document, config={})

    paragraphs = [block.text for block in result.text_blocks if block.kind == "paragraph"]
    assert any("multiple spaces and line breaks." in text for text in paragraphs)
    assert all("  " not in text for text in paragraphs)


def test_markdown_parser_can_handle_by_filename_hint() -> None:
    blob = _inline_blob_from_text("Sample content", media_type="text/plain")
    document = _DocumentStub(
        media_type="text/plain", blob=blob, origin_uri=None, name="notes.md"
    )
    parser = MarkdownDocumentParser()

    assert parser.can_handle(document)

    plain = _DocumentStub(media_type="text/plain", blob=blob, origin_uri=None)
    assert not parser.can_handle(plain)
