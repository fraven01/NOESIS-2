from __future__ import annotations

import json
from types import SimpleNamespace

from ai_core.tasks import cleanup_dead_letter_queue
from ai_core.tasks import alert_dead_letter_queue


class _FakeRedis:
    def __init__(self, items: list[bytes]) -> None:
        self.items = list(items)

    def lpop(self, _key: str):
        if not self.items:
            return None
        return self.items.pop(0)

    def llen(self, _key: str) -> int:
        return len(self.items)

    def rpush(self, _key: str, value: bytes) -> int:
        self.items.append(value)
        return len(self.items)


def test_cleanup_skips_non_redis_broker(monkeypatch) -> None:
    monkeypatch.setattr(
        "ai_core.tasks.settings",
        SimpleNamespace(CELERY_BROKER_URL="amqp://guest@localhost//"),
    )

    result = cleanup_dead_letter_queue()
    payload = result["data"]

    assert payload["status"] == "skipped"
    assert payload["reason"] == "non_redis_broker"


def test_cleanup_skips_when_ttl_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "ai_core.tasks.settings",
        SimpleNamespace(
            CELERY_BROKER_URL="redis://localhost:6379/0", CELERY_DLQ_TTL_MS=0
        ),
    )

    result = cleanup_dead_letter_queue()
    payload = result["data"]

    assert payload["status"] == "skipped"
    assert payload["reason"] == "ttl_disabled"


def test_cleanup_removes_expired_messages(monkeypatch) -> None:
    now = 100.0
    expired = json.dumps({"body": [{"dead_lettered_at": 90.0}]}).encode("utf-8")
    fresh = json.dumps({"body": [{"dead_lettered_at": 100.0}]}).encode("utf-8")
    invalid = b"not-json"
    fake_redis = _FakeRedis([expired, invalid, fresh])

    monkeypatch.setattr(
        "ai_core.tasks.settings",
        SimpleNamespace(
            CELERY_BROKER_URL="redis://localhost:6379/0", CELERY_DLQ_TTL_MS=1000
        ),
    )
    monkeypatch.setattr("ai_core.tasks.Redis.from_url", lambda _url: fake_redis)
    monkeypatch.setattr("ai_core.tasks.time.time", lambda: now)

    result = cleanup_dead_letter_queue(max_messages=10, ttl_ms=1000)
    payload = result["data"]

    assert payload["removed"] == 1
    assert payload["kept"] == 2
    assert expired not in fake_redis.items


def test_alert_emits_event_when_threshold_exceeded(monkeypatch) -> None:
    fake_redis = _FakeRedis([b"a", b"b", b"c"])
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        "ai_core.tasks.settings",
        SimpleNamespace(CELERY_BROKER_URL="redis://localhost:6379/0"),
    )
    monkeypatch.setattr("ai_core.tasks.Redis.from_url", lambda _url: fake_redis)
    monkeypatch.setattr(
        "ai_core.tasks.emit_event",
        lambda name, payload=None: events.append((name, payload or {})),
    )

    result = alert_dead_letter_queue(threshold=2)
    payload = result["data"]

    assert payload["alerted"] is True
    assert payload["queue_length"] == 3
    assert events
    assert events[0][0] == "dlq.threshold_exceeded"
