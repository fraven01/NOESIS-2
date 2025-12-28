import json

from ai_core import tasks
from ai_core.infra import object_store


def _write(tmp_path, relative_path: str, content: str) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_semantic_chunker_respects_section_boundaries(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path, raising=False)
    settings.RAG_CHUNK_TARGET_TOKENS = 64
    settings.RAG_CHUNK_OVERLAP_TOKENS = 0

    meta = {
        "scope_context": {
            "tenant_id": "tenant",
            "trace_id": "trace-1",
            "invocation_id": "invocation-1",
            "run_id": "run-1",
        },
        "business_context": {
            "case_id": "case",
        },
        "external_id": "ext",
        "title": "Doc",
    }
    text_path = "tenant/upload/text.md"
    _write(
        tmp_path,
        text_path,
        "Intro text. Detail text one. Detail text two. Outro text.",
    )

    blocks = [
        {"kind": "heading", "text": "Intro", "section_path": ["Intro"]},
        {"kind": "paragraph", "text": "Intro text.", "section_path": ["Intro"]},
        {
            "kind": "heading",
            "text": "Details",
            "section_path": ["Intro", "Details"],
        },
        {
            "kind": "paragraph",
            "text": "Detail text one.",
            "section_path": ["Intro", "Details"],
        },
        {
            "kind": "paragraph",
            "text": "Detail text two.",
            "section_path": ["Intro", "Details"],
        },
        {"kind": "heading", "text": "Outro", "section_path": ["Outro"]},
        {"kind": "paragraph", "text": "Outro text.", "section_path": ["Outro"]},
    ]
    blocks_path = "tenant/upload/blocks.json"
    _write(tmp_path, blocks_path, json.dumps({"blocks": blocks}))
    meta["parsed_blocks_path"] = blocks_path

    result = tasks.chunk(meta, text_path)
    payload = object_store.read_json(result["path"])
    contents = [chunk["content"] for chunk in payload["chunks"]]

    assert any("Intro / Details" in content for content in contents)
    assert any(content.endswith("Outro text.") for content in contents)
    assert not any(
        "Intro text." in content and "Outro text." in content for content in contents
    )


def test_semantic_chunker_keeps_heading_parents(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path, raising=False)
    settings.RAG_CHUNK_TARGET_TOKENS = 64
    settings.RAG_CHUNK_OVERLAP_TOKENS = 0

    meta = {
        "scope_context": {
            "tenant_id": "tenant",
            "trace_id": "trace-2",
            "invocation_id": "invocation-2",
            "run_id": "run-2",
        },
        "business_context": {
            "case_id": "case",
        },
        "external_id": "ext-2",
        "title": "Doc Two",
    }
    text_path = "tenant/upload/solo.md"
    _write(tmp_path, text_path, "Solo heading only. Info text.")

    blocks = [
        {"kind": "heading", "text": "Solo", "section_path": ["Solo"]},
        {"kind": "heading", "text": "Info", "section_path": ["Info"]},
        {"kind": "paragraph", "text": "Info text.", "section_path": ["Info"]},
    ]
    blocks_path = "tenant/upload/solo_blocks.json"
    _write(tmp_path, blocks_path, json.dumps({"blocks": blocks}))
    meta["parsed_blocks_path"] = blocks_path

    result = tasks.chunk(meta, text_path)
    payload = object_store.read_json(result["path"])

    parents = payload["parents"]
    solo_parent = next(node for node in parents.values() if node.get("title") == "Solo")
    assert "Solo" in solo_parent.get("content", "")

    info_chunk = next(
        chunk for chunk in payload["chunks"] if "Info text." in chunk["content"]
    )
    assert len(info_chunk["meta"]["parent_ids"]) >= 2
