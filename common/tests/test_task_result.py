from __future__ import annotations

from datetime import datetime, timezone

from common.task_result import TaskResult


def test_task_result_serializes_datetime() -> None:
    completed_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = TaskResult(
        status="success",
        data={"value": 1},
        context_snapshot={"tenant_id": "tenant"},
        task_name="test-task",
        completed_at=completed_at,
    )

    payload = result.model_dump(mode="json")

    assert payload["completed_at"] == "2025-01-01T00:00:00Z"
    assert payload["status"] == "success"
    assert payload["data"]["value"] == 1


def test_task_result_round_trip() -> None:
    completed_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    payload = TaskResult(
        status="success",
        data={"path": "/tmp/out.json"},
        context_snapshot={"trace_id": "trace-1"},
        task_name="test-task",
        completed_at=completed_at,
    ).model_dump(mode="json")

    parsed = TaskResult.model_validate(payload)

    assert parsed.task_name == "test-task"
    assert parsed.data["path"] == "/tmp/out.json"
