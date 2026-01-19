import json

from ai_core.rag.schemas import RagResponse


def test_rag_response_schema_exports() -> None:
    schema = RagResponse.model_json_schema()
    assert "answer_markdown" in schema.get("properties", {})
    assert "reasoning" in schema.get("properties", {})


def test_rag_response_round_trip() -> None:
    payload = {
        "reasoning": {"analysis": "Because A", "gaps": ["Missing B"]},
        "answer_markdown": "Answer [S1]",
        "used_sources": [{"id": "s1", "label": "Source 1", "relevance_score": 0.9}],
        "suggested_followups": ["Follow up?"],
    }
    model = RagResponse.model_validate(payload)
    dumped = json.loads(model.model_dump_json())
    assert dumped == payload
