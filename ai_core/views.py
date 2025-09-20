from __future__ import annotations

import json
from uuid import uuid4

from django.conf import settings
from django.db import connection
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


def _resolve_tenant_id(request: HttpRequest) -> str | None:
    """Derive the active tenant identifier for the current request."""

    tenant_obj = getattr(request, "tenant", None)
    schema_name = getattr(tenant_obj, "schema_name", None)
    if not schema_name:
        schema_name = getattr(connection, "schema_name", None)

    if not schema_name:
        return None

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    if schema_name == public_schema:
        return None

    return schema_name


def _prepare_request(request: HttpRequest):
    tenant_header = request.headers.get("X-Tenant-ID")
    case_id = request.headers.get("X-Case-ID")

    tenant = _resolve_tenant_id(request)
    if not tenant:
        return None, JsonResponse({"detail": "tenant not resolved"}, status=400)

    if tenant_header and tenant_header != tenant:
        return None, JsonResponse({"detail": "tenant mismatch"}, status=400)

    if not case_id:
        return None, JsonResponse({"detail": "missing case header"}, status=400)

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
