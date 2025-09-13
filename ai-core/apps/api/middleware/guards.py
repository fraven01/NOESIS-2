"""Request guards for tenant and case enforcement."""

from __future__ import annotations

import uuid
from fastapi import Request
from fastapi.responses import JSONResponse

from ...infra import rate_limit, logging as log


async def assert_case_active(request: Request, call_next):
    """Ensure tenant and case headers exist and attach trace metadata."""
    if request.url.path in ("/health", "/ready"):
        return await call_next(request)

    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        return JSONResponse(status_code=400, content={"detail": "missing X-Tenant-ID"})

    case_id = request.headers.get("X-Case-ID")
    if not case_id:
        return JSONResponse(status_code=401, content={"detail": "missing X-Case-ID"})

    if not rate_limit.check(tenant_id):
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})

    trace_id = uuid.uuid4().hex
    log.set_trace_id(trace_id)
    request.state.meta = {"tenant": tenant_id, "case": case_id, "trace": trace_id}

    try:
        return await call_next(request)
    finally:
        log.set_trace_id(None)
