from unittest.mock import patch
import pytest
from django.conf import settings
from rest_framework.test import APIClient
from django.test import override_settings
from types import SimpleNamespace

REST_FRAMEWORK_OVERRIDES = {
    **settings.REST_FRAMEWORK,
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [],
}


@pytest.fixture
def api_client():
    client = APIClient()
    # Mocking user on the client handler is tricky without force_authenticate
    # But with Auth Classes disabled, we might not need it.
    return client


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


@override_settings(REST_FRAMEWORK=REST_FRAMEWORK_OVERRIDES)
@patch("llm_worker.views.TenantContext")
def test_run_task_requires_tenant_header(mock_tenant_context, db, api_client):
    mock_tenant_context.from_request.return_value = None

    response = api_client.post(
        "/api/llm/run/", data=_score_results_payload(), format="json"
    )
    assert response.status_code == 400
    assert "X-Tenant-ID header is required" in response.json()["detail"]


@override_settings(REST_FRAMEWORK=REST_FRAMEWORK_OVERRIDES)
@patch("llm_worker.views.TenantContext")
@patch("llm_worker.views.submit_worker_task")
def test_run_task_returns_success(mock_submit, mock_tenant_context, db, api_client):
    mock_tenant_context.from_request.return_value = SimpleNamespace(
        schema_name="tenant"
    )
    mock_submit.return_value = (
        {"task_id": "task-1", "result": {"evaluations": []}, "state": {}},
        True,
    )

    response = api_client.post(
        "/api/llm/run/",
        data=_score_results_payload(),
        format="json",
        headers={"X-Tenant-ID": "tenant", "X-Case-ID": "case", "X-Trace-ID": "trace"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "succeeded"
    assert data["task_id"] == "task-1"


@override_settings(REST_FRAMEWORK=REST_FRAMEWORK_OVERRIDES)
@patch("llm_worker.views.TenantContext")
@patch(
    "llm_worker.views.submit_worker_task", return_value=({"task_id": "task-1"}, False)
)
def test_run_task_returns_queue_on_timeout(
    mock_submit, mock_tenant_context, db, api_client
):
    mock_tenant_context.from_request.return_value = SimpleNamespace(
        schema_name="tenant"
    )

    response = api_client.post(
        "/api/llm/run/",
        data=_score_results_payload(),
        format="json",
        headers={"X-Tenant-ID": "tenant"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["status_url"].endswith("/api/llm/tasks/task-1/")
