"""Tests for task status polling API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest
from django.test import RequestFactory
from rest_framework import status

from common.task_result import TaskResult
from llm_worker.views import task_status
from users.tests.factories import UserFactory


@pytest.mark.django_db
def test_task_status_pending():
    """Test task status endpoint returns 202 for pending tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-123/")
    request.user = UserFactory()  # Authenticate
    task_id = "test-task-123"

    # Mock AsyncResult with PENDING state
    mock_result = MagicMock()
    mock_result.state = "PENDING"

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        # Execute
        response = task_status(request, task_id)

        # Assert
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.content
        import json

        data = json.loads(data)
        assert data["status"] == "queued"
        assert data["task_id"] == task_id
        assert data["state"] == "PENDING"


@pytest.mark.django_db
def test_task_status_success():
    """Test task status endpoint returns 200 with payload for successful tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-456/")
    request.user = UserFactory()  # Authenticate
    task_id = "test-task-456"

    # Mock AsyncResult with SUCCESS state and result payload
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    task_result = TaskResult(
        status="success",
        data={
            "state": {"updated": "state"},
            "result": {"answer": "This is the answer", "prompt_version": "v1"},
            "cost_summary": {"total_usd": 0.1235, "components": []},
        },
        context_snapshot={},
        task_name="llm_worker.tasks.run_graph",
        completed_at=datetime.now(timezone.utc),
    )
    mock_result.result = task_result.model_dump(mode="json")

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        # Execute
        response = task_status(request, task_id)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.content
        import json

        data = json.loads(data)
        assert data["status"] == "succeeded"
        assert data["task_id"] == task_id
        # Verify result payload was merged
        assert "result" in data
    assert data["result"]["answer"] == "This is the answer"
    assert data["cost_summary"]["total_usd"] == 0.1235


@pytest.mark.django_db
def test_task_status_task_result_partial():
    """Test task status endpoint handles partial TaskResult payloads."""
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-partial/")
    request.user = UserFactory()
    task_id = "test-task-partial"

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    task_result = TaskResult(
        status="partial",
        data={
            "result": {"answer": "Partial answer"},
            "cost_summary": {"total_usd": 0.01, "components": []},
        },
        context_snapshot={},
        task_name="llm_worker.tasks.run_graph",
        completed_at=datetime.now(timezone.utc),
    )
    mock_result.result = task_result.model_dump(mode="json")

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        response = task_status(request, task_id)

        assert response.status_code == status.HTTP_200_OK
        import json

        data = json.loads(response.content)
        assert data["status"] == "succeeded"
        assert data["task_id"] == task_id
        assert data["result_status"] == "partial"
        assert data["result"]["answer"] == "Partial answer"


@pytest.mark.django_db
def test_task_status_task_result_error():
    """Test task status endpoint maps TaskResult error to failed response."""
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-error/")
    request.user = UserFactory()
    task_id = "test-task-error"

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    task_result = TaskResult(
        status="error",
        data={"detail": "bad input"},
        context_snapshot={},
        task_name="llm_worker.tasks.run_graph",
        completed_at=datetime.now(timezone.utc),
        error="boom",
    )
    mock_result.result = task_result.model_dump(mode="json")

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        response = task_status(request, task_id)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        import json

        data = json.loads(response.content)
        assert data["status"] == "failed"
        assert data["task_id"] == task_id
        assert data["error"]["type"] == "task_result_error"
        assert data["error"]["message"] == "boom"
        assert data["error"]["data"]["detail"] == "bad input"


@pytest.mark.django_db
def test_task_status_failure():
    """Test task status endpoint returns 500 with error details for failed tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-789/")
    request.user = UserFactory()  # Authenticate
    task_id = "test-task-789"

    # Mock AsyncResult with FAILURE state and exception info
    mock_result = MagicMock()
    mock_result.state = "FAILURE"
    mock_result.info = ValueError("Invalid input data")

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        # Execute
        response = task_status(request, task_id)

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.content
        import json

        data = json.loads(data)
        assert data["status"] == "failed"
        assert data["task_id"] == task_id
        assert "error" in data
        assert data["error"]["type"] == "ValueError"
        assert "Invalid input data" in data["error"]["message"]


@pytest.mark.django_db
def test_task_status_started():
    """Test task status endpoint returns 202 for started tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-started/")
    request.user = UserFactory()  # Authenticate
    task_id = "test-task-started"

    # Mock AsyncResult with STARTED state
    mock_result = MagicMock()
    mock_result.state = "STARTED"

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        # Execute
        response = task_status(request, task_id)

        # Assert
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.content
        import json

        data = json.loads(data)
        assert data["status"] == "queued"
        assert data["task_id"] == task_id
        assert data["state"] == "STARTED"


@pytest.mark.django_db
def test_task_status_revoked():
    """Test task status endpoint returns 500 for revoked tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-revoked/")
    request.user = UserFactory()  # Authenticate
    task_id = "test-task-revoked"

    # Mock AsyncResult with REVOKED state
    mock_result = MagicMock()
    mock_result.state = "REVOKED"
    mock_result.info = None

    with patch("llm_worker.views.AsyncResult") as mock_async_result:
        mock_async_result.return_value = mock_result

        # Execute
        response = task_status(request, task_id)

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.content
        import json

        data = json.loads(data)
        assert data["status"] == "failed"
        assert data["task_id"] == task_id
        assert "error" in data
        assert data["error"]["type"] == "revoked"
