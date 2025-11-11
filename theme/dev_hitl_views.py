"""Developer-only HITL approval endpoints and views."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any

from django.conf import settings
from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseForbidden,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import render
from django.utils.encoding import iri_to_uri
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_GET, require_POST
from structlog.stdlib import get_logger

from .dev_hitl_store import store

logger = get_logger(__name__)

_DEV_TOKEN_HEADER = "_DEV_ONLY_"


def _feature_enabled() -> bool:
    return bool(getattr(settings, "DEV_FEATURE_HITL_UI", False))


def _require_feature_enabled() -> None:
    if not _feature_enabled():
        raise Http404("Developer HITL UI is disabled")


def _has_dev_token(request: HttpRequest, *, allow_query: bool = False) -> bool:
    header_value = request.headers.get(_DEV_TOKEN_HEADER, "").lower()
    if header_value == "true":
        return True
    if allow_query:
        token = request.GET.get("dev_token", "").lower()
        if token == "true":
            return True
    return False


@require_GET
@csrf_protect
def dev_hitl_page(request: HttpRequest, run_id: str | None = None) -> HttpResponse:
    """Render the developer HITL approval page."""

    _require_feature_enabled()

    requested_run = (run_id or request.GET.get("run_id") or "").strip()
    if not requested_run:
        requested_run = store.default_run_id()

    run_state = store.get(requested_run)
    initial_payload = run_state.serialize()

    context = {
        "run_id": requested_run,
        "initial_payload": initial_payload,
        "dev_token_header": _DEV_TOKEN_HEADER,
        "dev_token_value": "true",
    }
    logger.info("hitl.dev.page_rendered", run_id=requested_run)
    return render(request, "theme/dev_hitl.html", context)


@require_GET
def get_run_payload(request: HttpRequest, run_id: str) -> JsonResponse:
    """Return the synthetic run payload for the developer UI."""

    _require_feature_enabled()
    if not _has_dev_token(request):
        return HttpResponseForbidden("Missing dev token header")

    run_state = store.get(run_id)
    logger.info("hitl.dev.run_payload", run_id=run_id)
    return JsonResponse(run_state.serialize())


@require_POST
@csrf_protect
def approve_candidates(request: HttpRequest) -> JsonResponse:
    """Accept approval decisions for the mock run."""

    _require_feature_enabled()
    if not _has_dev_token(request):
        return HttpResponseForbidden("Missing dev token header")

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse(
            {"error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST.value
        )

    validation_error = _validate_submission_payload(payload)
    if validation_error:
        return JsonResponse(validation_error, status=HTTPStatus.BAD_REQUEST.value)

    run_state = store.get(payload["run_id"])
    submission_payload = {
        "approved_ids": payload.get("approved_ids", []),
        "rejected_ids": payload.get("rejected_ids", []),
        "custom_urls": payload.get("custom_urls", []),
    }

    response, _is_new = run_state.record_submission(submission_payload)
    return JsonResponse(response)


@require_GET
def progress_stream(request: HttpRequest, run_id: str) -> StreamingHttpResponse:
    """Provide Server-Sent Events for ingestion/coverage progress."""

    _require_feature_enabled()
    if not _has_dev_token(request, allow_query=True):
        return HttpResponseForbidden("Missing dev token header")

    run_state = store.get(run_id)
    logger.info("hitl.dev.progress_stream", run_id=run_id)

    def event_stream():
        for event in run_state.stream_events():
            yield _format_sse(event)

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _format_sse(event: dict[str, Any]) -> str:
    event_type = event.get("type", "message")
    payload = json.dumps(event.get("payload", {}))
    lines = [f"event: {event_type}", f"data: {payload}"]
    return "\n".join(lines) + "\n\n"


def _validate_submission_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return {"error": "invalid_payload"}

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return {"error": "missing_run_id"}

    def _expect_list(values: Any, key: str) -> list[str] | None:
        if values in (None, ""):
            return []
        if not isinstance(values, list):
            return None
        sanitized: list[str] = []
        for value in values:
            if isinstance(value, str):
                sanitized.append(value.strip())
            else:
                return None
        return sanitized

    approved = _expect_list(payload.get("approved_ids"), "approved_ids")
    rejected = _expect_list(payload.get("rejected_ids"), "rejected_ids")
    custom_urls = _expect_list(payload.get("custom_urls"), "custom_urls")

    if approved is None or rejected is None or custom_urls is None:
        return {"error": "invalid_array"}

    invalid_urls = [url for url in custom_urls if not _is_valid_custom_url(url)]
    if invalid_urls:
        return {"error": "invalid_custom_urls", "details": invalid_urls}

    payload["approved_ids"] = approved
    payload["rejected_ids"] = rejected
    payload["custom_urls"] = custom_urls
    return None


def _is_valid_custom_url(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return False
    try:
        iri_to_uri(value)
    except Exception:  # pragma: no cover - Django handles edge cases robustly
        return False
    return True


__all__ = [
    "dev_hitl_page",
    "get_run_payload",
    "approve_candidates",
    "progress_stream",
]
