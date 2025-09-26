"""Tests for the chaos reporting pytest plugin."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.pytester import Pytester
from _pytest.reports import TestReport

from tests.chaos import reporting

pytest_plugins = ("pytester",)


class _Unserialisable:
    """Custom type with a deterministic string representation."""

    def __str__(self) -> str:  # pragma: no cover - simple data holder
        return "custom-object-value"


class _DummyMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyItem:
    """Minimal stand-in for a pytest test item."""

    def __init__(self) -> None:
        self.keywords = {"chaos": True}
        self.user_properties: list[tuple[str, object]] = []

    def iter_markers(self):
        return [_DummyMarker("chaos"), _DummyMarker("slow")]


def test_reporting_plugin_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pytester: Pytester
) -> None:
    """Ensure the chaos reporting plugin persists chaos metadata as JSON."""

    log_root = tmp_path / "chaos-artifacts"
    monkeypatch.setattr(reporting, "CHAOS_LOG_ROOT", log_root, raising=False)

    config = pytester.parseconfig()
    reporting.pytest_configure(config)
    assert log_root.is_dir(), "pytest_configure should create the CHAOS_LOG_ROOT"

    item = _DummyItem()
    reporting.pytest_runtest_setup(item)

    nodeid = "tests/chaos/test_reporting_plugin.py::test_case"
    env_values = {"REDIS_DOWN": "1", "SQL_DOWN": "0"}
    reporting.CHAOS_ENV_REGISTRY[nodeid] = SimpleNamespace(values=env_values)

    user_properties = list(item.user_properties)
    user_properties.extend(
        [
            ("chaos_env", env_values),
            ("user_note", {"severity": "medium"}),
            ("unserialisable", _Unserialisable()),
        ]
    )

    report = TestReport(
        nodeid=nodeid,
        location=("tests/chaos/test_reporting_plugin.py", 1, "test_case"),
        keywords={"chaos": True},
        outcome="passed",
        longrepr=None,
        when="call",
        sections=(),
        duration=0.1,
        user_properties=user_properties,
    )

    try:
        reporting.pytest_runtest_logreport(report)
    finally:
        reporting.CHAOS_ENV_REGISTRY.pop(nodeid, None)

    artifacts = list(log_root.glob("*.json"))
    assert len(artifacts) == 1, "expected a single chaos artifact"

    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))

    assert payload["markers"] == ["chaos", "slow"]
    assert payload["metadata"]["chaos_env"] == env_values
    assert payload["metadata"]["user_note"] == {"severity": "medium"}
    assert payload["metadata"]["unserialisable"] == "custom-object-value"
    assert "chaos_markers" not in payload["metadata"], "chaos markers should be promoted to top-level markers"
