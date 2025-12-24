"""Tests for HTTP request normalization."""

from unittest.mock import Mock, patch

from django.http import HttpRequest
from django.test import TestCase

from ai_core.contracts.scope import ScopeContext
from ai_core.ids.http_scope import normalize_request


class TestNormalizeRequest(TestCase):
    def setUp(self):
        # Patch TenantContext where it is defined, so when it is imported inside the function, it gets the mock
        # We need to patch 'customers.tenant_context.TenantContext'
        self.patcher = patch("customers.tenant_context.TenantContext")
        self.mock_tenant_context = self.patcher.start()
        self.mock_tenant_context.from_request.return_value = None

    def tearDown(self):
        self.patcher.stop()

    def test_happy_path(self):
        """Test normalization with all headers present."""
        request = HttpRequest()
        request.META = {
            "HTTP_X_TENANT_ID": "test-tenant",
            "HTTP_X_CASE_ID": "case-123",
            "HTTP_X_TRACE_ID": "trace-456",
            "HTTP_X_INVOCATION_ID": "inv-789",
            "HTTP_X_RUN_ID": "run-abc",
            "HTTP_X_WORKFLOW_ID": "wf-xyz",
            "HTTP_IDEMPOTENCY_KEY": "idem-key",
            "HTTP_X_TENANT_SCHEMA": "schema-test",
        }

        # Mock TenantContext to return a tenant matching the header
        mock_tenant = Mock()
        mock_tenant.schema_name = "test-tenant"
        self.mock_tenant_context.from_request.return_value = mock_tenant

        scope = normalize_request(request)

        assert isinstance(scope, ScopeContext)
        assert scope.tenant_id == "test-tenant"
        assert scope.case_id == "case-123"
        assert scope.trace_id == "trace-456"
        assert scope.invocation_id == "inv-789"
        assert scope.run_id == "run-abc"
        assert scope.ingestion_run_id is None
        assert scope.workflow_id == "wf-xyz"
        assert scope.idempotency_key == "idem-key"
        assert scope.tenant_schema == "schema-test"

    def test_missing_tenant_id(self):
        """Test that missing tenant_id raises ValidationError (via ScopeContext)."""
        request = HttpRequest()
        request.META = {}

        from customers.tenant_context import TenantRequiredError

        with self.assertRaises(TenantRequiredError):
            normalize_request(request)

    def test_tenant_context_fallback(self):
        """Test fallback to TenantContext if header is missing."""
        request = HttpRequest()
        request.META = {
            "HTTP_X_CASE_ID": "case-123",  # case_id is mandatory
        }

        # Setup mock to return a tenant
        mock_tenant = Mock()
        mock_tenant.schema_name = "fallback-tenant"
        self.mock_tenant_context.from_request.return_value = mock_tenant

        scope = normalize_request(request)

        assert scope.tenant_id == "fallback-tenant"

    def test_both_run_ids_allowed(self):
        """Test that both run_id and ingestion_run_id can co-exist (Pre-MVP ID Contract)."""
        request = HttpRequest()
        request.META = {
            "HTTP_X_TENANT_ID": "t1",
            "HTTP_X_CASE_ID": "case-123",
            "HTTP_X_RUN_ID": "run-1",
            "HTTP_X_INGESTION_RUN_ID": "ing-1",
        }

        mock_tenant = Mock()
        mock_tenant.schema_name = "t1"
        self.mock_tenant_context.from_request.return_value = mock_tenant

        # Should NOT raise - both IDs are now allowed
        scope = normalize_request(request)

        assert scope.run_id == "run-1"
        assert scope.ingestion_run_id == "ing-1"

    def test_auto_generate_ids(self):
        """Test that IDs are auto-generated if missing."""
        request = HttpRequest()
        request.META = {
            "HTTP_X_TENANT_ID": "t1",
            "HTTP_X_CASE_ID": "case-123",  # case_id is mandatory
        }

        mock_tenant = Mock()
        mock_tenant.schema_name = "t1"
        self.mock_tenant_context.from_request.return_value = mock_tenant

        scope = normalize_request(request)

        assert scope.tenant_id == "t1"
        assert scope.trace_id is not None
        assert scope.invocation_id is not None
        assert scope.run_id is not None
        assert scope.ingestion_run_id is None

    def test_ingestion_run_id_preference(self):
        """Test that ingestion_run_id is used if present."""
        request = HttpRequest()
        request.META = {
            "HTTP_X_TENANT_ID": "t1",
            "HTTP_X_CASE_ID": "case-123",  # case_id is mandatory
            "HTTP_X_INGESTION_RUN_ID": "ing-1",
        }

        mock_tenant = Mock()
        mock_tenant.schema_name = "t1"
        self.mock_tenant_context.from_request.return_value = mock_tenant

        scope = normalize_request(request)

        assert scope.ingestion_run_id == "ing-1"
        assert scope.run_id is None
