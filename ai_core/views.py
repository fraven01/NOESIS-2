from __future__ import annotations

import json
from uuid import uuid4

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .graphs import info_intake, needs_mapping, scope_check, system_description
from .infra import rate_limit
from .infra.object_store import read_json, write_json
from .infra.resp import apply_std_headers


def assert_case_active(tenant: str, case_id: str) -> None:
    """Placeholder for future case activity checks."""
    return None


def _prepare_request(request: HttpRequest):
    tenant = request.headers.get("X-Tenant-ID")
    case_id = request.headers.get("X-Case-ID")
    if not tenant or not case_id:
        return None, JsonResponse({"detail": "missing headers"}, status=400)

    if not rate_limit.check(tenant):
        return None, JsonResponse({"detail": "rate limit"}, status=429)

    trace_id = uuid4().hex
    assert_case_active(tenant, case_id)
    meta = {"tenant": tenant, "case": case_id, "trace_id": trace_id}
    return meta, None


def _load_state(tenant: str, case_id: str) -> dict:
    try:
        return read_json(f"{tenant}/{case_id}/state.json")
    except FileNotFoundError:
        return {}


def _save_state(tenant: str, case_id: str, state: dict) -> None:
    write_json(f"{tenant}/{case_id}/state.json", state)


def _run_graph(request: HttpRequest, graph) -> JsonResponse:
    meta, error = _prepare_request(request)
    if error:
        return error

    state = _load_state(meta["tenant"], meta["case"])
    if request.body:
        try:
            payload = json.loads(request.body)
            if isinstance(payload, dict):
                state.update(payload)
        except json.JSONDecodeError:
            return JsonResponse({"detail": "invalid json"}, status=400)

    try:
        new_state, result = graph.run(state, meta)
    except ValueError as exc:
        return JsonResponse({"detail": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"detail": "internal error"}, status=500)

    _save_state(meta["tenant"], meta["case"], new_state)

    prompt_version = result.get("prompt_version")
    response = JsonResponse(result)
    return apply_std_headers(response, meta["trace_id"], prompt_version)


def ping(request: HttpRequest) -> JsonResponse:
    """Lightweight endpoint used to verify AI Core availability."""
    meta, error = _prepare_request(request)
    if error:
        return error
    response = JsonResponse({"ok": True})
    return apply_std_headers(response, meta["trace_id"])


@csrf_exempt
@require_POST
def intake(request: HttpRequest) -> JsonResponse:
    return _run_graph(request, info_intake)


@csrf_exempt
@require_POST
def scope(request: HttpRequest) -> JsonResponse:
    return _run_graph(request, scope_check)


@csrf_exempt
@require_POST
def needs(request: HttpRequest) -> JsonResponse:
    return _run_graph(request, needs_mapping)


@csrf_exempt
@require_POST
def sysdesc(request: HttpRequest) -> JsonResponse:
    return _run_graph(request, system_description)
