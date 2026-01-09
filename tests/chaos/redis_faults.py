"""Chaos tests for Redis fault injection scenarios."""

from __future__ import annotations

import os
from typing import Iterable

import pytest
from celery import chain, signature
from celery.canvas import Signature
from kombu.connection import Connection as KombuConnection
from kombu.exceptions import OperationalError
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from structlog.testing import capture_logs

import ai_core.infra.observability as observability
from ai_core.infra import rate_limit
from common import logging as common_logging
from common.celery import with_scope_apply_async
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
)
from tests.chaos.conftest import _build_chaos_meta

pytestmark = pytest.mark.chaos


def _redis_down() -> bool:
    """Return ``True`` when the REDIS_DOWN chaos flag is enabled."""

    return os.getenv("REDIS_DOWN", "0").lower() in {"1", "true", "yes"}


@pytest.fixture(autouse=True)
def redis_faults(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterable[None]:
    """Inject Redis and Celery broker failures when ``REDIS_DOWN`` is active."""

    if "chaos" not in request.node.keywords:
        # Only activate the Redis fault injection for chaos-marked tests. If this
        # module's fixtures are imported elsewhere without the marker, the
        # Celery signature patch would otherwise leak into unrelated suites.
        yield
        return

    original_incr = Redis.incr
    original_get = Redis.get
    original_set = Redis.set
    original_publish = Redis.publish
    original_connect = KombuConnection.connect
    original_apply_async = Signature.apply_async

    def _maybe_conn(self: Redis, *args: object, **kwargs: object):  # type: ignore[override]
        if _redis_down():
            raise RedisConnectionError("redis chaos: connection refused")
        return original_incr(self, *args, **kwargs)

    def _maybe_timeout_get(self: Redis, *args: object, **kwargs: object):  # type: ignore[override]
        if _redis_down():
            raise RedisTimeoutError("redis chaos: operation timed out")
        return original_get(self, *args, **kwargs)

    def _maybe_timeout_set(self: Redis, *args: object, **kwargs: object):  # type: ignore[override]
        if _redis_down():
            raise RedisTimeoutError("redis chaos: operation timed out")
        return original_set(self, *args, **kwargs)

    def _maybe_publish(self: Redis, *args: object, **kwargs: object):  # type: ignore[override]
        if _redis_down():
            raise RedisConnectionError("redis chaos: connection refused")
        return original_publish(self, *args, **kwargs)

    def _maybe_connect(self: KombuConnection, *args: object, **kwargs: object):  # type: ignore[override]
        if _redis_down():
            raise OperationalError("redis chaos: broker unreachable")
        return original_connect(self, *args, **kwargs)

    def _maybe_apply_async(self: Signature, *args: object, **kwargs: object):  # type: ignore[override]
        if _redis_down():
            raise OperationalError("redis chaos: broker publish failed")
        return original_apply_async(self, *args, **kwargs)

    monkeypatch.setattr(Redis, "incr", _maybe_conn, raising=False)
    monkeypatch.setattr(Redis, "get", _maybe_timeout_get, raising=False)
    monkeypatch.setattr(Redis, "set", _maybe_timeout_set, raising=False)
    monkeypatch.setattr(Redis, "publish", _maybe_publish, raising=False)
    monkeypatch.setattr(KombuConnection, "connect", _maybe_connect, raising=False)
    monkeypatch.setattr(Signature, "apply_async", _maybe_apply_async, raising=False)

    yield


@pytest.mark.django_db
def test_ping_graceful_degradation(
    client,
    chaos_env,
    test_tenant_schema_name,
    caplog,
    monkeypatch,
):
    """The ``/ai/ping/`` endpoint continues to succeed when Redis is down."""

    rate_limit.reset_cache()
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.local")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")
    monkeypatch.setenv("LITELLM_API_KEY", "token")
    chaos_env.set_redis_down(True)

    tenant = test_tenant_schema_name
    headers = {
        X_TENANT_SCHEMA_HEADER: tenant,
        X_TENANT_ID_HEADER: tenant,
        X_CASE_ID_HEADER: "chaos-ping",
        X_TRACE_ID_HEADER: "trace-chaos-ping",
        IDEMPOTENCY_KEY_HEADER: "chaos-ping-001",
    }

    with caplog.at_level("WARNING", logger="ai_core.infra.rate_limit"):
        response = client.get("/ai/ping/", **headers)

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response[X_TENANT_ID_HEADER] == tenant
    assert response[X_CASE_ID_HEADER] == "chaos-ping"
    assert IDEMPOTENCY_KEY_HEADER not in response
    assert any("fail-open" in record.message for record in caplog.records)


def _produce_agents_task(meta: dict[str, object]) -> bool:
    """Attempt to enqueue an agents task with new meta structure.

    Args:
        meta: Meta dict with scope_context and business_context (new compositional structure)

    Returns:
        True if task was successfully enqueued, False on Redis/broker failure
    """

    logger = common_logging.get_logger("tests.chaos.agents")

    # Extract IDs from new compositional structure
    scope_context = meta.get("scope_context", {})
    business_context = meta.get("business_context", {})
    if isinstance(scope_context, dict):
        tenant_id = scope_context.get("tenant_id")
        trace_id = scope_context.get("trace_id")
    else:
        tenant_id = None
        trace_id = None
    if isinstance(business_context, dict):
        case_id = business_context.get("case_id")
    else:
        case_id = None

    try:
        pipeline = chain(signature("ai_core.tasks.ingest_raw"))
        with_scope_apply_async(pipeline, meta)
        observability.emit_event(  # pragma: no cover - defensive success path
            {
                "event": "agents.queue.scheduled",
                "tenant": tenant_id,
                "case_id": case_id,
                "trace_id": trace_id,
            }
        )
        return True
    except (OperationalError, RedisConnectionError, RedisTimeoutError) as exc:
        observability.emit_event(
            {
                "event": "agents.queue.backoff",
                "tenant": tenant_id,
                "case_id": case_id,
                "trace_id": trace_id,
                "error": str(exc),
            }
        )
        logger.warning(
            "agents.queue.backoff",
            error=str(exc),
            tenant=tenant_id,
            case=case_id,
        )
        return False


@pytest.mark.django_db
def test_task_producer_backoff_logs_metrics(
    chaos_env,
    test_tenant_schema_name,
    monkeypatch,
):
    """Task producers log backoff and emit telemetry when Redis is down."""

    chaos_env.set_redis_down(True)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.local")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")
    monkeypatch.setenv("LITELLM_API_KEY", "token")

    captured_events: list[dict[str, object]] = []

    def _record(payload: dict[str, object]) -> None:
        captured_events.append(payload)

    monkeypatch.setattr(observability, "emit_event", _record)

    meta = _build_chaos_meta(
        tenant_id=test_tenant_schema_name,
        trace_id="chaos-trace",
        case_id="chaos-case",
        run_id="run-chaos-backoff",
    )

    with capture_logs() as logs:
        success = _produce_agents_task(meta)

    assert success is False
    assert any(
        entry.get("event") == "agents.queue.backoff" for entry in captured_events
    )
    assert any(log.get("event") == "agents.queue.backoff" for log in logs)


@pytest.mark.django_db
def test_agent_error_records_langfuse_tags(
    chaos_env,
    test_tenant_schema_name,
    langfuse_mock,
    monkeypatch,
):
    """Agent errors under Redis outages should emit Langfuse spans and tags."""

    chaos_env.set_redis_down(True)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.local")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")
    monkeypatch.setenv("LITELLM_API_KEY", "token")

    meta = _build_chaos_meta(
        tenant_id=test_tenant_schema_name,
        trace_id="chaos-agent-trace",
        case_id="chaos-agent-error",
        run_id="run-chaos-agent-error",
    )

    success = _produce_agents_task(meta)

    assert success is False
    assert langfuse_mock.sample_rate == "1.0"
    agent_spans = [
        span for span in langfuse_mock.spans if span.operation == "agents.run"
    ]

    assert agent_spans, "Expected agents.run span to be recorded for AgentError"
    assert any(
        span.metadata.get("error_type") == "redis.down" for span in agent_spans
    ), "Expected Langfuse span metadata to tag error_type=redis.down"
    assert any(
        span.metadata.get("tenant_id") == test_tenant_schema_name
        for span in agent_spans
    ), "Expected Langfuse span metadata to include tenant_id"
