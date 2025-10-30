"""Tests for the plain text fallback parser."""

from __future__ import annotations

from documents.parsers import ParserDispatcher, ParserRegistry
from documents.parsers_text import TextDocumentParser


class _BlobStub:
    def __init__(self, payload: bytes, media_type: str | None = None) -> None:
        self._payload = payload
        self.media_type = media_type
        self.content_encoding = None

    def decoded_payload(self) -> bytes:
        return self._payload


class _DocumentStub:
    def __init__(
        self,
        *,
        blob: _BlobStub,
        media_type: str | None = None,
        external_media_type: str | None = None,
    ) -> None:
        self.media_type = media_type
        self.blob = blob

        class _Meta:
            pass

        meta = _Meta()
        if external_media_type is None:
            meta.external_ref = None
        else:
            meta.external_ref = {"media_type": external_media_type}
        self.meta = meta


def test_text_parser_handles_explicit_text_media_type() -> None:
    blob = _BlobStub(b"First paragraph.\n\nSecond paragraph.", media_type="text/plain")
    document = _DocumentStub(blob=blob, media_type="text/plain")
    parser = TextDocumentParser()

    assert parser.can_handle(document) is True

    result = parser.parse(document, config={})

    assert len(result.text_blocks) == 2
    assert result.text_blocks[0].text == "First paragraph."
    assert result.text_blocks[1].text == "Second paragraph."
    assert result.statistics["parser.kind"] == "text/plain"
    assert result.statistics["parser.characters"] == len(
        "First paragraph.\n\nSecond paragraph."
    )


def test_text_parser_sniffs_utf8_payload_without_media_type() -> None:
    blob = _BlobStub(b"Intro text\n\nMore text", media_type=None)
    document = _DocumentStub(blob=blob, media_type=None, external_media_type=None)
    parser = TextDocumentParser()

    assert parser.can_handle(document) is True

    result = parser.parse(document, config={})

    assert [block.text for block in result.text_blocks] == ["Intro text", "More text"]
    assert result.statistics["parser.kind"] == "text/plain"


def test_dispatcher_uses_text_parser_as_fallback() -> None:
    class _RejectingParser:
        def can_handle(
            self, document: object
        ) -> bool:  # pragma: no cover - simple guard
            return False

        def parse(
            self, document: object, config: object
        ) -> object:  # pragma: no cover - unused
            raise AssertionError("should not be called")

    registry = ParserRegistry([_RejectingParser()])
    registry.register(TextDocumentParser())
    dispatcher = ParserDispatcher(registry)

    blob = _BlobStub(b"Solo paragraph", media_type=None)
    document = _DocumentStub(blob=blob)

    result = dispatcher.parse(document, config={})

    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].text == "Solo paragraph"
