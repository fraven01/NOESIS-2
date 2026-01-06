"""Performance smoke probes executed via pytest.

These mini-load tests serve as an early warning before heavier load-testing
stacks (k6/Locust) are run. They follow the QA smoke checklists to ensure the
error budget (5%) is respected and latency percentiles are recorded for the
staging gate.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List

import pytest
from django.db import close_old_connections, connections
from django.test import Client

from ai_core.infra import object_store, rate_limit
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    META_CASE_ID_KEY,
    META_IDEMPOTENCY_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    X_CASE_ID_HEADER,
    X_TENANT_ID_HEADER,
)

pytestmark = [pytest.mark.chaos, pytest.mark.perf_smoke]

_STAGING_WEB_CONCURRENCY = 30
_FAILURE_THRESHOLD = 0.05


@dataclass
class RagQueryResponse:
    """Result container for concurrent RAG query requests."""

    ok: bool
    latency_s: float
    status: int


def _percentile(values: Iterable[float], percentile: float) -> float:
    """Return the percentile using linear interpolation."""

    data: List[float] = sorted(values)
    if not data:
        raise ValueError("no data available for percentile calculation")
    if percentile <= 0:
        return data[0]
    if percentile >= 1:
        return data[-1]
    position = (len(data) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[int(position)]
    lower_value = data[lower]
    upper_value = data[upper]
    weight = position - lower
    return lower_value * (1 - weight) + upper_value * weight


def _execute_rag_query(
    case_id: str, tenant: str, tenant_schema: str
) -> RagQueryResponse:
    """Issue a single /v1/ai/rag/query/ request and record metrics."""

    client = Client()
    payload = {"question": "Ping?", "filters": {"topic": "chaos"}}
    headers = {
        META_TENANT_ID_KEY: tenant,
        META_TENANT_SCHEMA_KEY: tenant_schema,
        META_CASE_ID_KEY: case_id,
        META_IDEMPOTENCY_KEY: f"rag-chaos-{case_id}",
    }
    start = time.perf_counter()
    try:
        response = client.post(
            "/v1/ai/rag/query/",
            data=json.dumps(payload),
            content_type="application/json",
            **headers,
        )
    finally:
        close_old_connections()
    latency_s = time.perf_counter() - start
    ok = response.status_code == 200
    return RagQueryResponse(ok=ok, latency_s=latency_s, status=response.status_code)


@pytest.mark.django_db
def test_rag_query_perf_smoke(
    chaos_env,
    test_tenant_schema_name,
    monkeypatch,
    record_property,
    tmp_path,
):
    """Run a mini load probe for /v1/ai/rag/query/ mirroring QA abort criteria."""

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id not in {"gw0", "master"}:
        pytest.skip(
            "perf_smoke suite is limited to a dedicated worker to avoid Django connection contention"
        )

    chaos_env.reset()
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path / "object-store")

    tenant = test_tenant_schema_name
    tenant_schema = test_tenant_schema_name

    with ThreadPoolExecutor(max_workers=_STAGING_WEB_CONCURRENCY) as executor:
        futures = [
            executor.submit(_execute_rag_query, f"RQ{index:03d}", tenant, tenant_schema)
            for index in range(_STAGING_WEB_CONCURRENCY)
        ]
        results = [future.result() for future in as_completed(futures)]

    connections.close_all()

    latencies = [result.latency_s for result in results if result.ok]
    failures = [result for result in results if not result.ok]

    assert results, "expected rag responses to be recorded"
    assert latencies, "expected successful rag responses for percentile metrics"

    p50_ms = _percentile(latencies, 0.50) * 1000
    p95_ms = _percentile(latencies, 0.95) * 1000
    failure_rate = len(failures) / len(results)

    record_property("rag_perf_total", len(results))
    record_property("rag_perf_failures", len(failures))
    record_property("rag_perf_failure_rate", failure_rate)
    record_property("rag_perf_p50_ms", round(p50_ms, 3))
    record_property("rag_perf_p95_ms", round(p95_ms, 3))

    assert (
        failure_rate < _FAILURE_THRESHOLD
    ), "QA checklist abort: failure rate exceeded 5% allowance"

    # Document the key headers to aid debugging if a percentile spike occurs
    sample = results[0]
    record_property("rag_perf_first_status", sample.status)
    record_property(
        "rag_perf_headers",
        [X_TENANT_ID_HEADER, X_CASE_ID_HEADER, IDEMPOTENCY_KEY_HEADER],
    )
