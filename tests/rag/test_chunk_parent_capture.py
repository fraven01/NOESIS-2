from __future__ import annotations

import textwrap

import pytest

from ai_core import tasks
from ai_core.infra import object_store


@pytest.fixture
def configured_store(tmp_path, monkeypatch):
    base_path = tmp_path / "object-store"
    monkeypatch.setattr(object_store, "BASE_PATH", base_path)
    return base_path


def _run_chunk(meta: dict[str, str], markdown: str) -> dict[str, str]:
    raw = tasks.ingest_raw(meta, "doc.md", markdown.encode("utf-8"))
    text = tasks.extract_text(meta, raw["path"])
    return tasks.chunk(meta, text["path"])


def test_chunk_parent_contents_are_direct(settings, configured_store):
    del configured_store
    settings.RAG_PARENT_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_BYTES = 0
    settings.RAG_PARENT_CAPTURE_MAX_DEPTH = 0

    meta = {"tenant_id": "tenant", "case_id": "case", "external_id": "doc-direct"}
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

    root_id = f"{meta['external_id']}#doc"
    root_parent = parents[root_id]
    assert root_parent["content"] == "Document introduction outside sections."
    assert "Parent section text" not in root_parent["content"]

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

    meta = {"tenant_id": "tenant", "case_id": "case", "external_id": "doc-limits"}
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
