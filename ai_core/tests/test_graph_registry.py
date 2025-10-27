"""Tests for the in-memory graph runner registry."""

from __future__ import annotations

import types

import pytest

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

    registry.register("retrieval_augmented_generation", first)
    registry.register("retrieval_augmented_generation", second)

    assert registry.get("retrieval_augmented_generation") is second


def test_register_requires_name(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    runner = types.SimpleNamespace()

    with pytest.raises(ValueError, match="graph name must be provided"):
        registry.register("", runner)
