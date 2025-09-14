from ai_core.infra.prompts import load


def test_load_finds_prompt_and_version():
    data = load("retriever/answer")
    assert data["version"] == "v1"
    assert "Beantworte die Frage faktenbasiert" in data["text"]
