from __future__ import annotations

import textwrap
import uuid

import pytest

from ai_core import tasks
from ai_core.infra import object_store


@pytest.fixture
def configured_store(tmp_path, monkeypatch):
    base_path = tmp_path / "object-store"
    monkeypatch.setattr(object_store, "BASE_PATH", base_path)
    return base_path


def _build_scope_meta(doc_id: str, external_id: str) -> dict[str, str]:
    return {
        "scope_context": {
            "tenant_id": "tenant",
            "case_id": "case",
            "trace_id": str(uuid.uuid4()),
            "invocation_id": str(uuid.uuid4()),
            "run_id": str(uuid.uuid4()),
        },
        "external_id": external_id,
        "document_id": doc_id,
    }


def _run_chunk(meta: dict[str, object], markdown: str) -> dict[str, str]:
    raw = tasks.ingest_raw(meta, "doc.md", markdown.encode("utf-8"))
    text = tasks.extract_text(meta, raw["path"])
    return tasks.chunk(meta, text["path"])


def _run_chunk_with_blocks(
    meta: dict[str, object], markdown: str, blocks: list[dict[str, object]]
) -> dict[str, str]:
    raw = tasks.ingest_raw(meta, "doc.md", markdown.encode("utf-8"))
    text = tasks.extract_text(meta, raw["path"])
    blocks_path = tasks._build_path(meta, "text", "doc.parsed.json")
    object_store.write_json(
        blocks_path,
        {
            "text": markdown,
            "statistics": {"parser.kind": "markdown"},
            "blocks": blocks,
        },
    )
    meta["parsed_blocks_path"] = blocks_path
    return tasks.chunk(meta, text["path"])


def test_chunk_parent_contents_are_direct(settings, configured_store):
    del configured_store
    settings.RAG_PARENT_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_DEPTH = 0

    doc_id = str(uuid.uuid4())
    meta = _build_scope_meta(doc_id, "doc-direct")
    markdown = textwrap.dedent(
        """
        Document introduction outside sections.

        # Heading One
        Parent section text that should stay local.

        ## Child Section
        Child section body that must not leak upward.

        ### Grandchild
        Details that belong only to the grandchild section.

        ## Second Child
        Separate branch text.
        """
    ).strip()

    result = _run_chunk(meta, markdown)
    payload = object_store.read_json(result["path"])
    parents = payload["parents"]

    root_id = f"{doc_id}#doc"
    root_parent = parents[root_id]
    expected_root_content = textwrap.dedent(
        """
        Document introduction outside sections.

        # Heading One

        Parent section text that should stay local.

        ## Child Section

        Child section body that must not leak upward.

        ### Grandchild

        Details that belong only to the grandchild section.

        ## Second Child

        Separate branch text.
        """
    ).strip()
    assert root_parent["content"] == expected_root_content
    assert root_parent.get("document_id") == doc_id

    def _parent_by_title(title: str) -> dict[str, object]:
        for node in parents.values():
            if node.get("title") == title:
                return node
        raise AssertionError(f"Missing parent node for {title!r}")

    heading_parent = _parent_by_title("Heading One")
    assert "Child section body" not in heading_parent["content"]
    assert heading_parent["content"].startswith("# Heading One")

    child_parent = _parent_by_title("Child Section")
    assert "Grandchild" not in child_parent["content"]
    assert "Child section body" in child_parent["content"]

    grandchild_parent = _parent_by_title("Grandchild")
    assert "Details that belong only" in grandchild_parent["content"]
    assert "Separate branch text" not in grandchild_parent["content"]

    second_child_parent = _parent_by_title("Second Child")
    assert "Separate branch text" in second_child_parent.get("content", "")
    assert "Grandchild" not in second_child_parent.get("content", "")


def test_chunk_parent_capture_respects_limits(settings, configured_store):
    del configured_store
    settings.RAG_PARENT_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_BYTES = 120
    settings.RAG_PARENT_CAPTURE_MAX_DEPTH = 2

    doc_id = str(uuid.uuid4())
    meta = _build_scope_meta(doc_id, "doc-limits")
    level_one_body = " ".join(["Level one text"] * 40)
    level_two_body = " ".join(["Level two text"] * 40)
    level_three_body = " ".join(["Level three text"] * 40)
    markdown = textwrap.dedent(
        f"""
        # Level One
        {level_one_body}

        ## Level Two
        {level_two_body}

        ### Level Three
        {level_three_body}
        """
    ).strip()

    result = _run_chunk(meta, markdown)
    payload = object_store.read_json(result["path"])
    parents = payload["parents"]

    root_parent = parents[f"{doc_id}#doc"]
    expected_root_content = textwrap.dedent(
        f"""
        # Level One

        {level_one_body}

        ## Level Two

        {level_two_body}

        ### Level Three

        {level_three_body}
        """
    ).strip()
    assert root_parent["content"] == expected_root_content
    assert root_parent.get("capture_limited") is not True
    assert root_parent.get("document_id") == doc_id

    def _parent_by_title(title: str) -> dict[str, object]:
        for node in parents.values():
            if node.get("title") == title:
                return node
        raise AssertionError(f"Missing parent node for {title!r}")

    level_one_parent = _parent_by_title("Level One")
    assert len(level_one_parent["content"].encode("utf-8")) <= 120
    assert level_one_parent.get("capture_limited") is True

    level_two_parent = _parent_by_title("Level Two")
    assert len(level_two_parent["content"].encode("utf-8")) <= 120
    assert level_two_parent.get("capture_limited") is True

    level_three_parent = _parent_by_title("Level Three")
    assert "content" not in level_three_parent or not level_three_parent["content"]


def test_structured_chunk_flushes_when_heading_changes(settings, configured_store):
    del configured_store
    settings.RAG_PARENT_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_DEPTH = 0

    doc_id = str(uuid.uuid4())
    meta = _build_scope_meta(doc_id, "doc-structured-heading")
    markdown = textwrap.dedent(
        """
        # Section One

        First section body.

        # Section Two

        Second section body.
        """
    ).strip()

    blocks: list[dict[str, object]] = [
        {
            "index": 0,
            "kind": "heading",
            "text": "Section One",
            "section_path": ["Section One"],
        },
        {
            "index": 1,
            "kind": "paragraph",
            "text": "First section body.",
            "section_path": ["Section One"],
        },
        {
            "index": 2,
            "kind": "heading",
            "text": "Section Two",
            "section_path": ["Section Two"],
        },
        {
            "index": 3,
            "kind": "paragraph",
            "text": "Second section body.",
            "section_path": ["Section Two"],
        },
    ]

    result = _run_chunk_with_blocks(meta, markdown, blocks)
    payload = object_store.read_json(result["path"])
    chunks = payload["chunks"]
    parents = payload["parents"]

    def _parent_id(title: str) -> str:
        for identifier, info in parents.items():
            if info and info.get("title") == title:
                return identifier
        raise AssertionError(f"Missing parent with title {title!r}")

    section_one_parent = _parent_id("Section One")
    section_two_parent = _parent_id("Section Two")

    section_one_chunk = next(
        chunk for chunk in chunks if "First section body." in chunk["content"]
    )
    assert section_one_parent in section_one_chunk["meta"]["parent_ids"]
    assert "Second section body." not in section_one_chunk["content"]
    assert section_one_chunk["content"].splitlines()[0] == "Section One"

    section_two_chunk = next(
        chunk for chunk in chunks if "Second section body." in chunk["content"]
    )
    assert section_two_parent in section_two_chunk["meta"]["parent_ids"]
    assert "First section body." not in section_two_chunk["content"]
    assert section_two_chunk["content"].splitlines()[0] == "Section Two"


def test_structured_chunk_flushes_when_section_path_changes(settings, configured_store):
    del configured_store
    settings.RAG_PARENT_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_DEPTH = 0

    doc_id = str(uuid.uuid4())
    meta = _build_scope_meta(doc_id, "doc-section-path")
    markdown = textwrap.dedent(
        """
        # Section Alpha

        Alpha body.

        Beta body.
        """
    ).strip()

    blocks: list[dict[str, object]] = [
        {
            "index": 0,
            "kind": "heading",
            "text": "Section Alpha",
            "section_path": ["Section Alpha"],
        },
        {
            "index": 1,
            "kind": "paragraph",
            "text": "Alpha body.",
            "section_path": ["Section Alpha"],
        },
        {
            "index": 2,
            "kind": "paragraph",
            "text": "Beta body.",
            "section_path": ["Section Beta"],
        },
    ]

    result = _run_chunk_with_blocks(meta, markdown, blocks)
    payload = object_store.read_json(result["path"])
    chunks = payload["chunks"]
    parents = payload["parents"]

    def _parent_id(title: str) -> str:
        for identifier, info in parents.items():
            if info and info.get("title") == title:
                return identifier
        raise AssertionError(f"Missing parent with title {title!r}")

    alpha_parent = _parent_id("Section Alpha")
    beta_parent = _parent_id("Section Beta")

    alpha_chunk = next(chunk for chunk in chunks if "Alpha body." in chunk["content"])
    assert alpha_parent in alpha_chunk["meta"]["parent_ids"]
    assert "Beta body." not in alpha_chunk["content"]

    beta_chunk = next(chunk for chunk in chunks if "Beta body." in chunk["content"])
    assert beta_parent in beta_chunk["meta"]["parent_ids"]
    assert "Alpha body." not in beta_chunk["content"]


def test_fallback_flushes_pending_text_at_document_end(settings, configured_store):
    del configured_store
    settings.RAG_PARENT_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_DEPTH = 0

    doc_id = str(uuid.uuid4())
    meta = _build_scope_meta(doc_id, "doc-fallback-flush")
    markdown = textwrap.dedent(
        """
        Paragraph one without headings.

        Paragraph two closing the document.
        """
    ).strip()

    result = _run_chunk(meta, markdown)
    payload = object_store.read_json(result["path"])
    chunks = payload["chunks"]

    assert chunks, "expected at least one chunk from fallback path"
    combined_content = "\n\n".join(chunk["content"] for chunk in chunks)
    assert "Paragraph one without headings." in combined_content
    assert "Paragraph two closing the document." in combined_content
