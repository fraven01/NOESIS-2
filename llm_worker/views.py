"""Views for polling task status via Result Backend."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from celery.result import AsyncResult
from django.http import JsonResponse
from django.urls import reverse
from pydantic import ValidationError
from rest_framework import serializers, status
from rest_framework.decorators import api_view
from structlog.stdlib import get_logger

from llm_worker.runner import submit_worker_task
from llm_worker.schemas import WorkerTask
from noesis2.api import default_extend_schema
from noesis2.api.serializers import IdempotentResponseSerializer

logger = get_logger(__name__)


class WorkerRunCompletedResponseSerializer(IdempotentResponseSerializer):
    status = serializers.CharField()
    task_id = serializers.CharField()
    result = serializers.JSONField(required=False)
    state = serializers.JSONField(required=False)
    cost_summary = serializers.JSONField(required=False, allow_null=True)


class WorkerRunQueuedResponseSerializer(IdempotentResponseSerializer):
    status = serializers.CharField()
    task_id = serializers.CharField()
    status_url = serializers.URLField()


def _format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue.get("loc", ()))
        message = issue.get("msg", "Invalid input")
        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)
    return "; ".join(messages)


@default_extend_schema(
    summary="Submit a worker graph task",
    responses={
        200: WorkerRunCompletedResponseSerializer,
        202: WorkerRunQueuedResponseSerializer,
    },
)
@api_view(["POST"])
def run_task(request) -> JsonResponse:
    """Submit a worker task (currently score_results) and optionally wait for completion."""

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse(
            {"detail": "Invalid JSON payload"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        worker_task = WorkerTask.model_validate(payload)
    except ValidationError as exc:
        return JsonResponse(
            {"detail": _format_validation_error(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if worker_task.task_type != "score_results":
        return JsonResponse(
            {"detail": "Unsupported task_type. Only score_results is available."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        return JsonResponse(
            {"detail": "X-Tenant-ID header is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    case_id = request.headers.get("X-Case-ID") or "local"
    trace_id = request.headers.get("X-Trace-ID") or uuid4().hex

    meta_payload = worker_task.model_dump(mode="python")

    try:
        task_payload, completed = submit_worker_task(
            task_payload=meta_payload,
            scope={
                "tenant_id": tenant_id,
                "case_id": case_id,
                "trace_id": trace_id,
            },
            graph_name="score_results",
        )
    except Exception:
        logger.exception("llm_worker.run_task.failed", task_type=worker_task.task_type)
        return JsonResponse(
            {"detail": "Task submission failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if completed:
        return JsonResponse(
            {
                "idempotent": False,
                "status": "succeeded",
                "task_id": task_payload.get("task_id"),
                "result": task_payload.get("result"),
                "state": task_payload.get("state"),
                "cost_summary": task_payload.get("cost_summary"),
            },
            status=status.HTTP_200_OK,
        )

    status_url = request.build_absolute_uri(
        reverse("llm_worker:task_status", args=[task_payload["task_id"]])
    )
    return JsonResponse(
        {
            "idempotent": False,
            "status": "queued",
            "task_id": task_payload["task_id"],
            "status_url": status_url,
        },
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["GET"])
def task_status(request, task_id: str) -> JsonResponse:
    """
    Poll the status of a Celery task via Result Backend.

    This endpoint is used for polling tasks that returned 202 Accepted
    due to timeout in the graph worker pattern.

    State Mapping:
    - PENDING/RECEIVED/STARTED/RETRY → 202 Accepted (queued)
    - SUCCESS → 200 OK (succeeded with result payload)
    - FAILURE/REVOKED → 500 Internal Server Error (failed with error details)

    Args:
        request: Django request object
        task_id: Celery task ID

    Returns:
        JsonResponse with task status and metadata
    """
    async_result = AsyncResult(task_id)

    # Celery task states
    state = async_result.state

    # Map Celery states to API responses
    if state in ("PENDING", "RECEIVED", "STARTED", "RETRY"):
        # Task is still processing
        return JsonResponse(
            {
                "status": "queued",
                "task_id": task_id,
                "state": state,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    elif state == "SUCCESS":
        # Task completed successfully - return result payload
        try:
            result_payload = async_result.result
            if isinstance(result_payload, dict):
                # Merge result payload into response
                response_data: dict[str, Any] = {
                    "status": "succeeded",
                    "task_id": task_id,
                }
                response_data.update(result_payload)
                return JsonResponse(response_data, status=status.HTTP_200_OK)
            else:
                # Result is not a dict, wrap it
                return JsonResponse(
                    {
                        "status": "succeeded",
                        "task_id": task_id,
                        "result": result_payload,
                    },
                    status=status.HTTP_200_OK,
                )
        except Exception as exc:
            # Error retrieving result
            return JsonResponse(
                {
                    "status": "failed",
                    "task_id": task_id,
                    "error": {
                        "type": "result_retrieval_error",
                        "message": str(exc),
                    },
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    elif state in ("FAILURE", "REVOKED"):
        # Task failed or was revoked
        error_info: dict[str, str] = {
            "type": state.lower(),
            "message": "Task failed or was revoked",
        }

        # Try to get exception details
        try:
            if state == "FAILURE" and async_result.info:
                if isinstance(async_result.info, Exception):
                    error_info["message"] = str(async_result.info)
                    error_info["type"] = type(async_result.info).__name__
                elif isinstance(async_result.info, dict):
                    error_info.update(async_result.info)
        except Exception:
            # Ignore errors when retrieving exception info
            pass

        return JsonResponse(
            {
                "status": "failed",
                "task_id": task_id,
                "error": error_info,
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    else:
        # Unknown state - treat as queued
        return JsonResponse(
            {
                "status": "queued",
                "task_id": task_id,
                "state": state,
            },
            status=status.HTTP_202_ACCEPTED,
        )
