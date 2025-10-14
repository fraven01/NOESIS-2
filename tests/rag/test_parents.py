from ai_core.rag.parents import limit_parent_payload


def test_limit_parent_payload_no_cap(settings):
    settings.RAG_PARENT_MAX_BYTES = 0
    parents = {
        "section-1": {"id": "section-1", "content": "Short text"},
    }

    limited = limit_parent_payload(parents)

    assert limited["section-1"]["content"] == "Short text"
    assert "content_truncated" not in limited["section-1"]


def test_limit_parent_payload_applies_cap(settings):
    settings.RAG_PARENT_MAX_BYTES = 10
    text = "Lorem ipsum dolor sit amet"
    parents = {
        "section-1": {"id": "section-1", "content": text},
    }

    limited = limit_parent_payload(parents)
    node = limited["section-1"]

    assert len(node["content"].encode("utf-8")) <= 10
    assert node["content_truncated"] is True
    assert node["content_length"] == len(text)
    assert node["content_bytes"] == len(text.encode("utf-8"))
