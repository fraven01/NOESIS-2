import json
from unittest.mock import patch

from django.test import RequestFactory

from llm_worker.views import run_task


def _make_request(payload: dict, **headers) -> object:
    factory = RequestFactory()
    return factory.post(
        "/api/llm/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )


def _score_results_payload() -> dict:
    return {
        "task_type": "score_results",
        "control": {"model_preset": "demo"},
        "data": {
            "query": "test",
            "results": [
                {"id": "a", "title": "A", "snippet": "text", "url": "https://a"}
            ],
        },
    }


def test_run_task_requires_tenant_header():
    request = _make_request(_score_results_payload())
    response = run_task(request)
    assert response.status_code == 400


@patch("llm_worker.views.submit_worker_task")
def test_run_task_returns_success(mock_submit):
    mock_submit.return_value = (
        {"task_id": "task-1", "result": {"ranked": []}, "state": {}},
        True,
    )
    request = _make_request(
        _score_results_payload(),
        HTTP_X_TENANT_ID="tenant",
        HTTP_X_CASE_ID="case",
        HTTP_X_TRACE_ID="trace",
    )
    response = run_task(request)
    data = json.loads(response.content)
    assert response.status_code == 200
    assert data["status"] == "succeeded"
    assert data["task_id"] == "task-1"


@patch(
    "llm_worker.views.submit_worker_task", return_value=({"task_id": "task-1"}, False)
)
def test_run_task_returns_queue_on_timeout(mock_submit):
    request = _make_request(
        _score_results_payload(),
        HTTP_X_TENANT_ID="tenant",
    )
    response = run_task(request)
    data = json.loads(response.content)
    assert response.status_code == 202
    assert data["status"] == "queued"
    assert data["status_url"].endswith("/api/llm/tasks/task-1/")
