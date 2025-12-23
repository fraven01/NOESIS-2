import unittest
from unittest.mock import MagicMock, patch
from common.celery import ContextTask, with_scope_apply_async
from celery.canvas import Signature


class TestContextPropagation(unittest.TestCase):
    def setUp(self):
        # Patch OTel availability to True for these tests
        self.otel_patcher = patch("common.celery._OTEL_AVAILABLE", True)
        self.otel_patcher.start()

        # Patch OTel components - create=True needed if not present in module
        self.propagator_patcher = patch(
            "common.celery.TraceContextTextMapPropagator", create=True
        )
        self.MockPropagator = self.propagator_patcher.start()

        self.otel_context_patcher = patch("common.celery.otel_context", create=True)
        self.mock_otel_context = self.otel_context_patcher.start()

    def tearDown(self):
        try:
            self.otel_patcher.stop()
            self.propagator_patcher.stop()
            self.otel_context_patcher.stop()
        except RuntimeError:
            pass

    def test_producer_injection(self):
        """Test that with_scope_apply_async injects current trace headers."""
        # Setup mock signature
        mock_sig = MagicMock(spec=Signature)
        mock_sig.clone.return_value = mock_sig
        mock_sig.options = {}
        mock_sig.tasks = []  # Not a chain
        mock_sig.body = None

        # Setup mock propagator to inject specific headers
        def inject_side_effect(carrier):
            carrier["traceparent"] = "00-test_trace_id-span_id-01"

        self.MockPropagator.return_value.inject.side_effect = inject_side_effect

        # Call
        scope = {"tenant_id": "t1"}
        with_scope_apply_async(mock_sig, scope)

        # Verify headers were injected into signature options
        self.assertIn("headers", mock_sig.options)
        self.assertEqual(
            mock_sig.options["headers"]["traceparent"], "00-test_trace_id-span_id-01"
        )

        # Verify propagator inject was called
        self.MockPropagator.return_value.inject.assert_called_once()

    @patch("celery.Task.__call__")
    def test_consumer_extraction(self, mock_task_call):
        """Test that ContextTask extracts and activates trace headers."""
        mock_task_call.return_value = "ok"

        class TestTask(ContextTask):
            name = "test_task"

            def run(self, *args, **kwargs):
                return "ok"

        task = TestTask()
        task.request = MagicMock()  # Ensure request object exists
        task.request.headers = {"traceparent": "00-incoming-span-01"}

        # Setup Propagator extract
        mock_ctx = MagicMock()
        self.MockPropagator.return_value.extract.return_value = mock_ctx

        # Setup OTel attach
        mock_token = "token_123"
        self.mock_otel_context.attach.return_value = mock_token

        # Call the task (__call__ wrapper)
        result = task()

        # Verify extract called with headers
        self.MockPropagator.return_value.extract.assert_called_with(
            carrier=task.request.headers
        )

        # Verify attach called with extracted context
        self.mock_otel_context.attach.assert_called_with(mock_ctx)

        # Verify detach called with token
        self.mock_otel_context.detach.assert_called_with(mock_token)

        self.assertEqual(result, "ok")

    @patch("celery.Task.__call__")
    def test_no_otel_graceful_degradation(self, mock_task_call):
        """Test graceful degradation when OTel is not available."""
        mock_task_call.return_value = "ok"
        with patch("common.celery._OTEL_AVAILABLE", False):
            # Producer should not crash
            mock_sig = MagicMock(spec=Signature)
            mock_sig.clone.return_value = mock_sig
            mock_sig.options = {}  # No headers
            mock_sig.tasks = []
            mock_sig.body = None

            with_scope_apply_async(mock_sig, {})
            # Should have no headers injected (except maybe existing options)
            self.assertNotIn("headers", mock_sig.options)

            # Consumer should not crash
            class TestTask(ContextTask):
                name = "test_task"

                def run(self):
                    return "ok"

            task = TestTask()
            task.request = MagicMock()
            task.request.headers = {"traceparent": "something"}
            task()

            # Propagator should NOT be called
            self.MockPropagator.return_value.extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
