from __future__ import annotations

import pytest

from documents.parsers_image import ImageDocumentParser


class _BlobStub:
    def __init__(self, payload: bytes, media_type: str | None) -> None:
        self.content = payload
        self.media_type = media_type

    def decoded_payload(self) -> bytes:
        return self.content


class _DocumentStub:
    def __init__(self, media_type: str | None, blob: _BlobStub) -> None:
        self.media_type = media_type
        self.blob = blob


@pytest.mark.parametrize(
    "media_type",
    [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/tiff",
        "image/bmp",
    ],
)
def test_image_parser_can_handle_supported_media_types(media_type: str) -> None:
    parser = ImageDocumentParser()
    document = _DocumentStub(media_type=media_type, blob=_BlobStub(b"data", media_type))

    assert parser.can_handle(document) is True


@pytest.mark.parametrize(
    "media_type, blob_media_type",
    [
        ("image/svg+xml", None),
        (None, "application/octet-stream"),
        (None, None),
    ],
)
def test_image_parser_rejects_unsupported_media_types(
    media_type: str | None, blob_media_type: str | None
) -> None:
    parser = ImageDocumentParser()
    document = _DocumentStub(
        media_type=media_type, blob=_BlobStub(b"data", blob_media_type)
    )

    assert parser.can_handle(document) is False


def test_image_parser_returns_placeholder_asset_and_text_block() -> None:
    payload = b"\x89PNG\r\n\x1a\n"
    parser = ImageDocumentParser()
    document = _DocumentStub(
        media_type="image/png", blob=_BlobStub(payload, "image/png")
    )

    result = parser.parse(document, config={})

    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.media_type == "image/png"
    assert asset.content == payload
    assert dict(asset.metadata) == {
        "asset_kind": "document_content",
        "locator": "image-body",
    }

    assert len(result.text_blocks) == 1
    placeholder = result.text_blocks[0]
    assert placeholder.text == "[Bilddatei]"
    assert placeholder.kind == "other"

    assert result.statistics["parser.kind"] == "image"
