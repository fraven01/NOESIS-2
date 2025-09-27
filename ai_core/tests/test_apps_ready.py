"""Smoke tests for the AiCore application configuration."""

from __future__ import annotations

from django.apps import apps as django_apps


def test_app_ready_invokes_bootstrap(monkeypatch):
    calls: list[str] = []

    def _bootstrap():
        calls.append("called")

    monkeypatch.setattr("ai_core.graph.bootstrap.bootstrap", _bootstrap)

    config = django_apps.get_app_config("ai_core")
    config.ready()
    config.ready()

    assert calls == ["called", "called"]
