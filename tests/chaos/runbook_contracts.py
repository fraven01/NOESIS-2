"""Contract tests asserting incident runbook coverage and behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest
from structlog.testing import capture_logs

from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)
from kombu.exceptions import OperationalError
from tests.chaos.redis_faults import _produce_agents_task
from tests.chaos import redis_faults as redis_faults_module

pytestmark = pytest.mark.chaos

RUNBOOK_PATH = Path("docs/runbooks/incidents.md")

RUNBOOK_SCENARIOS: Iterable[tuple[str, tuple[str, ...]]] = (
    (
        "Memorystore/Redis down",
        (
            "Worker pausieren",
            "Connector-Status prüfen",
            "Failover-Redis aktivieren",
        ),
    ),
    (
        "Cloud SQL nicht erreichbar",
        (
            "Traffic einfrieren",
            "Statusseite aktualisieren",
            "Verbindungslimits prüfen",
        ),
    ),
    (
        "HTTP 5xx Spike",
        (
            "Traffic-Rollback",
            "letzte stabile Revision aktivieren",
            "Ressourcen-Auslastung prüfen",
        ),
    ),
)


@pytest.mark.parametrize("scenario_label, actions", RUNBOOK_SCENARIOS)
def test_runbook_documents_core_incident_scenarios(
    scenario_label: str, actions: tuple[str, ...]
):
    """Ensure the incident runbook lists required scenarios and first actions."""

    runbook_text = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert scenario_label in runbook_text
    for action in actions:
        assert action in runbook_text


@pytest.mark.django_db
def test_redis_outage_behaviour_matches_runbook(
    client,
    chaos_env,
    monkeypatch: pytest.MonkeyPatch,
    test_tenant_schema_name: str,
) -> None:
    """Redis outages should surface the runbook guidance in behaviour and logs."""

    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.local")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")
    monkeypatch.setenv("LITELLM_API_KEY", "token")

    chaos_env.set_redis_down(True)

    def _raise_operational_error(*args: object, **kwargs: object) -> None:
        raise OperationalError("redis chaos: broker unreachable")

    monkeypatch.setattr(
        redis_faults_module,
        "with_scope_apply_async",
        _raise_operational_error,
    )

    scope = {
        "tenant_id": test_tenant_schema_name,
        "case_id": "incident-redis",  # mirrors incident trace metadata
        "trace_id": "incident-redis-trace",
    }

    with capture_logs() as logs:
        success = _produce_agents_task(scope)

    assert success is False, "task producers must pause when Redis is unavailable"

    backoff_logs = [log for log in logs if log.get("event") == "agents.queue.backoff"]
    assert (
        len(backoff_logs) == 1
    ), "workers should not flood retries during Redis outages"

    runbook_hints = [
        log
        for log in logs
        if "Worker pausieren" in " ".join(str(value) for value in log.values())
        or "Connector-Status" in " ".join(str(value) for value in log.values())
    ]
    assert runbook_hints, "logs must reference incident runbook first actions"

    headers = {
        META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "incident-redis",
    }

    response = client.get("/health/", **headers)
    assert response.status_code == 503

    payload = response.json()
    assert payload.get("status") != "ok"

    services = payload.get("services", {})
    redis_status = services.get("redis")
    assert isinstance(redis_status, str)
    assert any(
        keyword in redis_status.lower() for keyword in ("degraded", "down", "paused")
    )
