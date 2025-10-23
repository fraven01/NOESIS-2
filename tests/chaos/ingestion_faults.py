"""Chaos tests for ingestion worker rate limits and deduplication."""

from __future__ import annotations

from types import SimpleNamespace
import uuid

import pytest
from celery.exceptions import Retry

import ai_core.infra.observability as observability
from ai_core import tasks
from ai_core.llm import client as llm_client
from ai_core.rag import vector_client

pytestmark = pytest.mark.chaos
pytest_plugins = ["tests.chaos.fixtures", "ai_core.tests.conftest"]


def _rate_limit_reader(_path: str):
    raise llm_client.RateLimitError("rate limited", status=429)


def test_ingestion_embed_retry_profile_and_dead_letter(monkeypatch):
    """Embedding task should back off exponentially and flag dead letters."""

    embed_task = tasks.embed
    expected_delays = [30, 60, 120, 240, 300]
    recorded_delays: list[int | None] = []
    observed_events: list[dict[str, object]] = []
    meta = {
        "tenant_id": "tenant-chaos",
        "case_id": "case-chaos",
        "trace_id": "trace-chaos",
    }
    request = SimpleNamespace(retries=0, headers={}, kwargs={"meta": meta})

    monkeypatch.setattr(embed_task, "request", request, raising=False)
    monkeypatch.setattr(tasks.object_store, "read_json", _rate_limit_reader)

    def _fake_retry(self, *_, exc=None, countdown=None, **__):
        recorded_delays.append(countdown)
        attempt = getattr(self.request, "retries", 0)
        if attempt >= self.max_retries:
            raise self.MaxRetriesExceededError("max retries exceeded")
        raise Retry(exc=exc, when=countdown)

    monkeypatch.setattr(embed_task, "retry", _fake_retry, raising=False)

    def _record_event(payload: dict[str, object]) -> None:
        observed_events.append(payload)

    monkeypatch.setattr(observability, "emit_event", _record_event)

    for attempt, expected in enumerate(expected_delays):
        request.retries = attempt
        with pytest.raises(Retry):
            embed_task(meta, "embeddings/chunks.json")
        assert recorded_delays[-1] == expected

    request.retries = len(expected_delays)
    with pytest.raises(embed_task.MaxRetriesExceededError):
        embed_task(meta, "embeddings/chunks.json")

    assert any(
        event.get("event") == "ingestion.dead_letter" for event in observed_events
    )
    assert recorded_delays == expected_delays


@pytest.mark.usefixtures("rag_database")
def test_ingestion_upsert_hash_prevents_duplicates(tmp_path, monkeypatch):
    """Repeated ingestion of the same payload must not duplicate vectors."""

    monkeypatch.chdir(tmp_path)
    vector_client.reset_default_client()

    tenant = str(uuid.uuid4())
    case = str(uuid.uuid4())
    meta = {"tenant_id": tenant, "case_id": case}

    def _run_pipeline() -> int:
        raw = tasks.ingest_raw(meta, "doc.txt", b"User 123")
        text = tasks.extract_text(meta, raw["path"])
        masked = tasks.pii_mask(meta, text["path"])
        chunks = tasks.chunk(meta, masked["path"])
        embeds = tasks.embed(meta, chunks["path"])
        return tasks.upsert(meta, embeds["path"])

    first = _run_pipeline()
    second = _run_pipeline()

    client = vector_client.get_default_client()
    results = client.search("User", {"tenant_id": tenant, "case_id": case})

    assert first == 1
    assert second == 1
    assert len(results) == 1
    assert results[0].content == "User XXX"

    vector_client.reset_default_client()
