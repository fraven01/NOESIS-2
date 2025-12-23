"""Tests for model usage reporting and telemetry."""

from unittest.mock import MagicMock, patch

import pytest
from ai_core.infra.observability import report_generation_usage
from ai_core.infra.usage import Usage


@pytest.fixture
def mock_span():
    """Mock the current OTel span."""
    with patch("ai_core.infra.observability._get_current_span") as mock:
        span = MagicMock()
        mock.return_value = span
        yield span


def test_usage_addition():
    """Verify Usage object accumulation."""
    u1 = Usage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01)
    u2 = Usage(input_tokens=20, output_tokens=10, total_tokens=30, cost_usd=0.02)
    total = u1 + u2
    
    assert total.input_tokens == 30
    assert total.output_tokens == 15
    assert total.total_tokens == 45
    assert abs(total.cost_usd - 0.03) < 1e-9


def test_usage_from_provider_response():
    """Verify Usage creation from provider response."""
    # OpenAI generic style
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    
    usage = Usage.from_provider_response(mock_response)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150

    # Dict style
    usage_dict = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    }
    usage = Usage.from_provider_response(usage_dict)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15


def test_report_generation_usage(mock_span):
    """Verify usage reporting sets correct span attributes."""
    usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.005)
    
    report_generation_usage(usage, model="gpt-4")
    
    # Verify set_attribute calls
    calls = mock_span.set_attribute.call_args_list
    attributes = {c[0][0]: c[0][1] for c in calls}
    
    assert attributes["gen_ai.usage.input_tokens"] == 10
    assert attributes["gen_ai.usage.output_tokens"] == 20
    assert attributes["gen_ai.usage.total_tokens"] == 30
    assert attributes["gen_ai.usage.cost"] == 0.005
    assert attributes["gen_ai.request.model"] == "gpt-4"
