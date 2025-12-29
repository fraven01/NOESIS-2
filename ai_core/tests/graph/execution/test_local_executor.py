"""Tests for the local graph executor."""

from unittest.mock import Mock

import pytest

from ai_core.graph.execution.local import LocalGraphExecutor
from ai_core.graph import registry


class TestLocalGraphExecutor:
    
    def test_run_delegates_to_registry(self):
        """Test that run() looks up the graph and calls its run method."""
        mock_runner = Mock()
        mock_runner.run.return_value = ({"out": 1}, {"result": "ok"})
        
        # Register the mock
        registry.register("test_graph", mock_runner)
        
        executor = LocalGraphExecutor()
        state, result = executor.run("test_graph", {"in": 1}, {"meta": "data"})
        
        # Verify call
        mock_runner.run.assert_called_once_with({"in": 1}, {"meta": "data"})
        assert state == {"out": 1}
        assert result == {"result": "ok"}

    def test_submit_raises_not_implemented(self):
        """Test that submit() raises NotImplementedError."""
        executor = LocalGraphExecutor()
        with pytest.raises(NotImplementedError):
            executor.submit("any_graph", {}, {})
