import pytest
from django.test import RequestFactory
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock

from customers.tests.factories import TenantFactory
from theme.views import ingestion_submit


@pytest.mark.django_db
def test_ingestion_submit_happy_path(settings):
    # Ensure RAG_UPLOAD_ENABLED is True if checked, but it seems not checked involved in view

    tenant_schema = "test_ingestion"
    tenant = TenantFactory(schema_name=tenant_schema)
    # The view expects request.tenant to be set by middleware

    factory = RequestFactory()
    file_content = b"fake content"
    uploaded_file = SimpleUploadedFile(
        "test.txt", file_content, content_type="text/plain"
    )

    request = factory.post(
        reverse("ingestion-submit"),
        data={"file": uploaded_file, "case_id": "case-123"},
        format="multipart",  # RequestFactory handles this slightly differently than Client
    )
    request.tenant = tenant

    # We mock handle_document_upload to isolate view logic first
    # If the view logic itself wraps/extracts wrongly, this might fail or we might see
    # the failure in _resolve_manual_collection which happens BEFORE handle_document_upload

    with patch("ai_core.services.handle_document_upload") as mock_handle:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.data = {
            "ingestion_run_id": "run-123",
            "decision": "ingest",
            "reason": "Because I said so",
        }
        mock_handle.return_value = mock_response

        response = ingestion_submit(request)

        assert response.status_code == 200
        assert (
            b"Ingestion started" in response.content or b"ingest" in response.content
        )  # Check partial content

        # Verify call args
        mock_handle.assert_called_once()
        args, kwargs = mock_handle.call_args
        meta = kwargs.get("meta") or args[2]

        print(f"DEBUG: meta passed to service: {meta}")

        assert "scope_context" in meta
        assert meta["scope_context"]["tenant_id"] == tenant.schema_name
        assert meta["scope_context"]["collection_id"] is not None


@pytest.mark.django_db
def test_ingestion_submit_real_service_call(settings):
    """Call the real service stack to see if vector client logging issues appear."""
    tenant_schema = "test_ingestion_real"
    tenant = TenantFactory(schema_name=tenant_schema)

    factory = RequestFactory()
    file_content = b"real content"
    uploaded_file = SimpleUploadedFile(
        "real.txt", file_content, content_type="text/plain"
    )

    request = factory.post(
        reverse("ingestion-submit"),
        data={"file": uploaded_file},
    )
    request.tenant = tenant

    # We DO NOT mock handle_document_upload here.
    # We DO mock the UNIVERSAL INGESTION GRAPH invocation to avoid full Celery/Graph execution complexity
    # effectively testing View -> Service glue -> Persistence -> (stop before graph)

    with patch(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph"
    ) as mock_build_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "output": {"decision": "ingest", "reason": "ok"}
        }
        mock_build_graph.return_value = mock_graph

        # We also need to mock _enqueue_ingestion_task to avoid Celery
        with patch("ai_core.services._enqueue_ingestion_task"):
            response = ingestion_submit(request)

            assert response.status_code == 200
            # If this passes, the issue is elusive.
            # If it fails with "ingestion_submit.failed", we captured it!
