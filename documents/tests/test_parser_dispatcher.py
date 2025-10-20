"""Tests for parser contracts and dispatcher behaviour."""

from __future__ import annotations

import pytest

from documents.parsers import (
    ParsedAsset,
    ParsedResult,
    ParsedTextBlock,
    ParserDispatcher,
    ParserRegistry,
)


class _FakeParser:
    def __init__(self, media_type: str) -> None:
        self.media_type = media_type
        self.handled: list[dict[str, str]] = []

    def can_handle(self, document: dict[str, str]) -> bool:
        return document.get("media_type") == self.media_type

    def parse(self, document: dict[str, str], config: object) -> ParsedResult:
        self.handled.append(document)
        return ParsedResult(
            text_blocks=(
                ParsedTextBlock(text=f"parsed-{self.media_type}", kind="paragraph"),
            ),
            assets=(),
            statistics={"media_type": self.media_type},
        )


def test_dispatcher_selects_first_matching_parser() -> None:
    pdf_parser = _FakeParser("application/pdf")
    docx_parser = _FakeParser("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    registry = ParserRegistry([docx_parser, pdf_parser])
    dispatcher = ParserDispatcher(registry)

    document = {"media_type": "application/pdf"}
    result = dispatcher.parse(document, config={})

    assert pdf_parser.handled == [document]
    assert docx_parser.handled == []
    assert result.statistics["media_type"] == "application/pdf"
    assert result.statistics["parse.blocks.total"] == 1
    assert result.statistics["parse.assets.total"] == 0


def test_dispatcher_raises_when_no_parser_found() -> None:
    registry = ParserRegistry()
    registry.register(_FakeParser("application/pdf"))
    dispatcher = ParserDispatcher(registry)

    with pytest.raises(RuntimeError, match="no_parser_found"):
        dispatcher.parse({"media_type": "text/plain"}, config={})


def test_parsed_text_block_schema_validation() -> None:
    block = ParsedTextBlock(
        text="Heading",
        kind="heading",
        section_path=[" Section 1 ", "Subsection\u200b"],
        page_index=2,
        table_meta={"rows": 4},
        language="en",
    )

    assert block.section_path == ("Section 1", "Subsection")
    assert block.page_index == 2
    assert block.table_meta == {"rows": 4}
    assert block.language == "en"

    with pytest.raises(TypeError):
        block.table_meta["rows"] = 5  # type: ignore[index]

    with pytest.raises(ValueError, match="parsed_text_block_kind"):
        ParsedTextBlock(text="Invalid", kind="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="parsed_text_block_page_index"):
        ParsedTextBlock(text="Text", kind="paragraph", page_index=-1)

    with pytest.raises(ValueError, match="parsed_text_block_section_path"):
        ParsedTextBlock(text="Text", kind="paragraph", section_path=["   "])

    with pytest.raises(ValueError, match="parsed_text_block_section_path"):
        ParsedTextBlock(text="Text", kind="paragraph", section_path=["a"] * 11)

    with pytest.raises(ValueError, match="parsed_text_block_section_path"):
        ParsedTextBlock(text="Text", kind="paragraph", section_path=["x" * 129])

    with pytest.raises(ValueError, match="parsed_text_block_language"):
        ParsedTextBlock(text="Text", kind="paragraph", language="invalid-lang-")


def test_parsed_asset_schema_validation() -> None:
    long_context = "x" * 600
    asset = ParsedAsset(
        media_type="image/png",
        content=b"bytes",
        page_index=0,
        bbox=[0.1, 0.2, 0.9, 0.95],
        context_before=long_context,
        context_after=long_context,
    )

    assert asset.content == b"bytes"
    assert asset.file_uri is None
    assert asset.bbox == (0.1, 0.2, 0.9, 0.95)
    assert len(asset.context_before or "") == 512
    assert len(asset.context_after or "") == 512

    with pytest.raises(ValueError, match="parsed_asset_location"):
        ParsedAsset(media_type="image/png")

    with pytest.raises(ValueError, match="parsed_asset_page_index"):
        ParsedAsset(media_type="image/png", content=b"x", page_index=-2)

    with pytest.raises(ValueError, match="parsed_asset_bbox_range"):
        ParsedAsset(media_type="image/png", content=b"x", bbox=[0, 0, 2, 1])

    with pytest.raises(ValueError, match="parsed_asset_bbox_range"):
        ParsedAsset(media_type="image/png", content=b"x", bbox=[0, 0, 1])


def test_parsed_result_wrangles_collections() -> None:
    block = ParsedTextBlock(text="Para", kind="paragraph")
    asset = ParsedAsset(media_type="image/jpeg", file_uri="s3://bucket/object.jpg")
    result = ParsedResult(
        text_blocks=[block], assets=[asset], statistics={"parser.pages": 2}
    )

    assert result.text_blocks == (block,)
    assert result.assets == (asset,)
    assert result.statistics["parser.pages"] == 2
    assert result.statistics["parse.blocks.total"] == 1
    assert result.statistics["parse.assets.total"] == 1

    with pytest.raises(ValueError, match="parsed_result_text_blocks"):
        ParsedResult(text_blocks=["invalid"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="parsed_result_statistics"):
        ParsedResult(statistics=[("pages", 1)])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="parsed_result_statistics"):
        ParsedResult(statistics={"pages": {1: 2}})

    with pytest.raises(ValueError, match="parsed_result_statistics"):
        ParsedResult(statistics={"weird": {"set": {1, 2}}})

    with pytest.raises(ValueError, match="parsed_result_statistics"):
        ParsedResult(statistics={"ratios": [float("nan")]})


def test_parser_registry_rejects_invalid_parsers() -> None:
    registry = ParserRegistry()

    class _MissingCanHandle:
        def parse(self, document: object, config: object) -> ParsedResult:  # pragma: no cover - invalid
            raise AssertionError

    class _MissingParse:
        def can_handle(self, document: object) -> bool:  # pragma: no cover - invalid
            return False

    with pytest.raises(TypeError, match="parser_missing_can_handle"):
        registry.register(_MissingCanHandle())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="parser_missing_parse"):
        registry.register(_MissingParse())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="parser_missing_can_handle"):
        registry.register(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="parser_invalid"):
        registry.register(None)  # type: ignore[arg-type]

