import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies BEFORE importing common.celery
sys.modules["celery"] = MagicMock()
sys.modules["celery.canvas"] = MagicMock()
sys.modules["ai_core.infra.pii_flags"] = MagicMock()
sys.modules["ai_core.infra.policy"] = MagicMock()
sys.modules["common.logging"] = MagicMock()

# We need common.constants to actually exist or be mocked if it's imported
# Assuming common.constants is simple enough or we mock it too
# But common.celery imports specific names.
sys.modules["common.constants"] = MagicMock()
sys.modules["common.constants"].HEADER_CANDIDATE_MAP = {"trace_id": ["X-Trace-Id"]}

# Mock opentelemetry BEFORE import if we want to test availability
# But common.celery uses try/except.
# Let's let it fail import (default) initially, or mock it.
# We want to test WITH OTel.
mock_otel = MagicMock()
mock_otel.trace.propagation.trace_context.TraceContextTextMapPropagator = MagicMock
sys.modules["opentelemetry"] = mock_otel
sys.modules["opentelemetry.trace"] = MagicMock()
sys.modules["opentelemetry.trace.propagation"] = MagicMock()
sys.modules["opentelemetry.trace.propagation.trace_context"] = MagicMock()

# Now import the target
# Since we are mocking everything, we need to manually define ContextTask and functions
# OR we rely on the file content.
# Better to import the actual file, but with mocks in place.
# We need to make sure the import sees our mocks.
import os

sys.path.append(os.getcwd())

from common.celery import ContextTask, with_scope_apply_async


class TestContextPropagationStandalone(unittest.TestCase):
    def setUp(self):
        # We need to ensure _OTEL_AVAILABLE is True for the test
        # It's a global in common.celery
        import common.celery

        common.celery._OTEL_AVAILABLE = True

        # Setup Propagator mock
        self.mock_propagator_cls = common.celery.TraceContextTextMapPropagator
        self.mock_propagator_cls.return_value = MagicMock()

        # Setup otel_context
        self.mock_otel_context = common.celery.otel_context

    def test_producer_injection(self):
        print("Testing Producer Injection...")
        mock_sig = MagicMock()
        mock_sig.clone.return_value = mock_sig
        mock_sig.options = {}
        mock_sig.tasks = []
        mock_sig.body = None

        # Setup Inject
        def inject(carrier):
            carrier["traceparent"] = "test-header"

        self.mock_propagator_cls.return_value.inject.side_effect = inject

        with_scope_apply_async(mock_sig, {"tenant_id": "t1"})

        if (
            "headers" in mock_sig.options
            and mock_sig.options["headers"].get("traceparent") == "test-header"
        ):
            print("PASS: Producer Injection")
        else:
            print(f"FAIL: Producer Injection. Options: {mock_sig.options}")
            self.fail("Headers not injected")

    def test_consumer_extraction(self):
        print("Testing Consumer Extraction...")

        class TestTask(ContextTask):
            name = "test"

            def run(self):
                return "ok"

        task = TestTask()
        task.request = MagicMock()
        task.request.headers = {"traceparent": "incoming"}

        self.mock_propagator_cls.return_value.extract.return_value = "extracted_ctx"
        self.mock_otel_context.attach.return_value = "token"

        task()

        self.mock_propagator_cls.return_value.extract.assert_called_with(
            carrier=task.request.headers
        )
        self.mock_otel_context.attach.assert_called_with("extracted_ctx")
        self.mock_otel_context.detach.assert_called_with("token")
        print("PASS: Consumer Extraction")


if __name__ == "__main__":
    unittest.main()
