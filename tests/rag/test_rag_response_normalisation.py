import uuid

from ai_core.views import _normalise_rag_response


def test_normalise_rag_response_serialises_nested_uuids() -> None:
    answer_key = uuid.uuid4()
    answer_value = uuid.uuid4()
    snippet_id = uuid.uuid4()
    retrieval_uuid_key = uuid.uuid4()
    retrieval_uuid_value = uuid.uuid4()

    payload = {
        "answer": {answer_key: answer_value},
        "prompt_version": "v1",
        "snippets": [{"id": snippet_id}],
        "retrieval": {
            "alpha": 0.3,
            "took_ms": 12.0,
            "vector_candidates": [
                {
                    "document_id": retrieval_uuid_key,
                    "metadata": {"source_id": retrieval_uuid_value},
                }
            ],
            "routing": {
                "profile": "default",
                "vector_space_id": uuid.uuid4(),
            },
        },
    }

    normalised = _normalise_rag_response(payload)

    answer = normalised["answer"]
    assert answer == {str(answer_key): str(answer_value)}

    snippets = normalised["snippets"]
    assert snippets == [{"id": str(snippet_id)}]

    retrieval = normalised["retrieval"]
    assert retrieval["took_ms"] == 12
    assert retrieval["vector_candidates"][0]["document_id"] == str(retrieval_uuid_key)
    assert (
        retrieval["vector_candidates"][0]["metadata"]["source_id"]
        == str(retrieval_uuid_value)
    )
    assert isinstance(retrieval["routing"], dict)
    assert retrieval["routing"]["profile"] == "default"
    assert isinstance(retrieval["routing"]["vector_space_id"], str)


def test_normalise_rag_response_collects_extras_in_diagnostics() -> None:
    extra_top_level_uuid = uuid.uuid4()
    extra_routing_uuid = uuid.uuid4()

    payload = {
        "answer": "ok",
        "prompt_version": "v1",
        "snippets": [],
        "retrieval": {
            "alpha": 0.1,
            "took_ms": 3.7,
            "unexpected": "value",
            "routing": {
                "profile": "default",
                "vector_space_id": "vs_123",
                "extra": extra_routing_uuid,
            },
        },
        "extra_field": {extra_top_level_uuid: "data"},
    }

    normalised = _normalise_rag_response(payload)

    assert "diagnostics" in normalised

    diagnostics = normalised["diagnostics"]
    assert diagnostics["retrieval"]["unexpected"] == "value"
    assert diagnostics["retrieval"]["routing"]["extra"] == str(extra_routing_uuid)
    assert diagnostics["response"]["extra_field"][str(extra_top_level_uuid)] == "data"
