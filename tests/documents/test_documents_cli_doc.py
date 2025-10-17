from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/documents/cli-howto.md")


def _load() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_cli_doc_exists() -> None:
    assert DOC_PATH.exists(), "CLI how-to document is missing"


def test_commands_covered() -> None:
    text = _load()
    tokens = [
        '"docs", "add"',
        '"docs", "get"',
        '"docs", "list"',
        '"docs", "delete"',
        '"assets", "add"',
        '"assets", "get"',
        '"assets", "list"',
        '"assets", "delete"',
    ]
    for token in tokens:
        assert token in text, f"command fragment {token!r} missing"


def test_media_type_hint_present() -> None:
    text = _load()
    assert "--media-type" in text
    assert "nur relevant" in text


def test_warning_documented() -> None:
    text = _load()
    assert "ocr_text_missing_for_ocr_only" in text


def test_schema_all_documented() -> None:
    text = _load()
    assert '"schema", "print"' in text
    assert '"--kind", "all"' in text


def test_troubleshooting_entries() -> None:
    text = _load()
    for code in [
        "blob_source_required",
        "base64_invalid",
        "storage_uri_missing",
        "validation_error",
        "document_not_found",
        "asset_not_found",
        "schema_kind_invalid",
        "ocr_text_missing_for_ocr_only",
    ]:
        assert f"`{code}`" in text, f"troubleshooting entry `{code}` missing"


def test_filter_example_present() -> None:
    text = _load()
    assert "jq '.items[] | select" in text
    assert "assets.json" in text
