from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/documents/contracts-reference.md")


def _load() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_reference_exists() -> None:
    assert DOC_PATH.exists(), "contracts reference document is missing"


def test_model_sections_present() -> None:
    text = _load()
    sections = [
        "## DocumentRef",
        "## DocumentMeta",
        "## BlobLocator",
        "### FileBlob",
        "### InlineBlob",
        "### ExternalBlob",
        "## NormalizedDocument",
        "## AssetRef",
        "## Asset",
        "## Hinweise",
    ]
    for section in sections:
        assert section in text, f"section {section!r} missing"


def test_anchor_links_available() -> None:
    text = _load()
    anchors = [
        "[DocumentRef](#documentref)",
        "[DocumentMeta](#documentmeta)",
        "[BlobLocator](#bloblocator)",
        "[Asset](#asset)",
    ]
    for anchor in anchors:
        assert anchor in text, f"anchor {anchor!r} missing"


def test_error_codes_documented() -> None:
    text = _load()
    expected_codes = [
        "tenant_empty",
        "uuid_invalid",
        "language_invalid",
        "sha256_invalid",
        "inline_size_mismatch",
        "literal_error",
        "checksum_invalid",
        "bbox_invalid",
        "caption_confidence_range",
        "workflow_empty",
        "workflow_invalid_char",
        "workflow_too_long",
        "meta_workflow_mismatch",
        "asset_workflow_mismatch",
    ]
    for code in expected_codes:
        assert f"`{code}`" in text, f"error code `{code}` missing"


def test_invalid_examples_cover_core_constraints() -> None:
    text = _load()
    for marker in [
        "version_invalid",
        "sha256_invalid",
        "language_invalid",
        "bbox_invalid",
    ]:
        assert marker in text, f"invalid example for {marker} missing"


def test_media_type_restriction_documented() -> None:
    text = _load()
    assert "media_type_invalid" in text
    assert "type/subtype" in text
