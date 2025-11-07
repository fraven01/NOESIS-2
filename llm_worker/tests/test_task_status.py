"""Tests for task status polling API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from django.test import RequestFactory
from rest_framework import status

from llm_worker.views import task_status


@pytest.mark.django_db
def test_task_status_pending():
    """Test task status endpoint returns 202 for pending tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-123/")
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
    task_id = "test-task-456"

    # Mock AsyncResult with SUCCESS state and result payload
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {
        "state": {"updated": "state"},
        "result": {"answer": "This is the answer", "prompt_version": "v1"},
        "cost_summary": {"total_usd": 0.1235, "components": []},
    }

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
def test_task_status_failure():
    """Test task status endpoint returns 500 with error details for failed tasks."""
    # Setup
    factory = RequestFactory()
    request = factory.get("/api/llm/tasks/test-task-789/")
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
