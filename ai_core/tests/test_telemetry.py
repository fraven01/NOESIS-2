from __future__ import annotations

from typing import Any, List, Mapping

import pytest

from ai_core import services


@pytest.fixture
def captured_events() -> List[Mapping[str, Any]]:
    events = []

    def emit_event(name: str, payload: Mapping[str, Any]) -> None:
        events.append({"name": name, "payload": payload})

    return events


@pytest.mark.django_db
def test_upload_telemetry(captured_events, monkeypatch):
    monkeypatch.setattr(services, "emit_event", captured_events.append)

    # ...


@pytest.mark.django_db
def test_crawl_telemetry(captured_events, monkeypatch):
    monkeypatch.setattr(services, "emit_event", captured_events.append)

    # ...
