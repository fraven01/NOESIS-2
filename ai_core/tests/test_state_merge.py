"""Tests for merging graph state payloads."""

from __future__ import annotations

from ai_core.graph.schemas import merge_state


def test_merge_state_overwrites_incoming_values() -> None:
    original = {"count": 1, "status": "pending", "nested": {"name": "old"}}
    incoming = {"count": 2, "nested": {"name": "new"}, "extra": True}

    merged = merge_state(original, incoming)

    assert merged["count"] == 2
    assert merged["status"] == "pending"
    assert merged["nested"] == {"name": "new"}
    assert merged["extra"] is True


def test_merge_state_handles_empty_inputs() -> None:
    assert merge_state(None, None) == {}
    assert merge_state({"foo": "bar"}, None) == {"foo": "bar"}
    assert merge_state(None, {"foo": "baz"}) == {"foo": "baz"}


def test_merge_state_preserves_missing_keys() -> None:
    original = {"retained": True, "value": 1}
    incoming = {"value": 2}

    merged = merge_state(original, incoming)

    assert merged["value"] == 2
    assert merged["retained"] is True
