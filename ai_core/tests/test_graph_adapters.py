"""Tests for the legacy module adapter."""

from __future__ import annotations

from types import ModuleType

import pytest

from ai_core.graph.adapters import module_runner


def test_module_runner_requires_callable_run() -> None:
    module = ModuleType("ai_core.graphs.stub")

    with pytest.raises(AttributeError) as excinfo:
        module_runner(module)

    message = str(excinfo.value)
    assert "ai_core.graphs.stub" in message
    assert "callable 'run'" in message
