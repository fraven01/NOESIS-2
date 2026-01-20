from ai_core.infra.prompts import load


def test_load_finds_prompt_and_version():
    data = load("retriever/answer")
    assert data["version"] == "v2"
    assert "<answer>" in data["text"]


def test_load_prefers_highest_numeric_version():
    data = load("testdata/multi")
    assert data["version"] == "v11"
    assert "Test prompt version 11" in data["text"]
