"""Tests for the in-memory graph runner registry."""

from __future__ import annotations

import types

from ai_core.graph import registry


def test_register_and_get_returns_registered_runner(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    runner = types.SimpleNamespace()

    registry.register("info_intake", runner)

    assert registry.get("info_intake") is runner


def test_register_overwrites_existing_runner(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    first = types.SimpleNamespace()
    second = types.SimpleNamespace()

    registry.register("scope_check", first)
    registry.register("scope_check", second)

    assert registry.get("scope_check") is second
