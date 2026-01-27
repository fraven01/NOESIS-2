from __future__ import annotations

import pytest

from ai_core.tests.fixtures.back_edge_graph import GRAPH
from scripts.lint_bounded_capabilities import validate_graph_edges


def test_bounded_linter_rejects_simple_back_edge_graph() -> None:
    with pytest.raises(ValueError, match="back_edge_detected"):
        validate_graph_edges(GRAPH)
