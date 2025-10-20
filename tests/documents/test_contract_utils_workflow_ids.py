import pytest

from documents.contract_utils import normalize_workflow_id


def test_normalize_workflow_id_strips_invisibles_and_whitespace():
    assert normalize_workflow_id("\u200b Workflow\u200d-ID \u00a0") == "Workflow-ID"


@pytest.mark.parametrize(
    "value,code",
    [
        ("", "workflow_empty"),
        ("   ", "workflow_empty"),
        ("\u200b\u200b", "workflow_empty"),
        ("a" * 129, "workflow_too_long"),
    ],
)
def test_normalize_workflow_id_rejects_empty_and_too_long(value, code):
    with pytest.raises(ValueError) as exc:
        normalize_workflow_id(value)
    assert str(exc.value) == code


def test_normalize_workflow_id_rejects_invalid_characters():
    with pytest.raises(ValueError) as exc:
        normalize_workflow_id("invalid id")
    assert str(exc.value) == "workflow_invalid_char"


def test_normalize_workflow_id_allows_valid_identifier():
    assert normalize_workflow_id(" ingest_2024-01 ") == "ingest_2024-01"
