import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from rest_framework.test import APIRequestFactory
from django.conf import settings

from documents.dev_api import DocumentDevViewSet

# from documents.models import Document  # Removed dependency
# from documents.tests.factories import DocumentFactory # Removed dependency


@pytest.fixture
def api_factory():
    return APIRequestFactory()


@pytest.fixture
def mock_vector_client():
    with patch("ai_core.rag.vector_client.get_default_client") as mock:
        yield mock.return_value


@pytest.fixture
def mock_tenant_context():
    with patch("documents.dev_api.TenantContext") as mock:
        mock_tenant = Mock()
        mock_tenant.id = uuid4()
        mock_tenant.schema_name = "public"
        mock.resolve_identifier.return_value = mock_tenant
        yield mock


@pytest.mark.django_db
def test_delete_document_hard_cleanup(
    api_factory, mock_vector_client, mock_tenant_context
):
    """Test that hard delete triggers vector cleanup."""
    # Enable DEBUG to allow dev API usage
    settings.DEBUG = True

    # Create a mock document
    doc = Mock()
    doc.id = uuid4()
    doc_id = str(doc.id)
    doc.delete = Mock()

    tenant_id = "test-tenant"

    view = DocumentDevViewSet.as_view({"delete": "delete_document"})
    request = api_factory.delete(f"/dev/{tenant_id}/{doc_id}/delete?hard=true")

    # Mock database object retrieval to return our mock doc
    with patch("documents.dev_api.get_object_or_404", return_value=doc):
        # Mock schema_context since we don't have real tenant schemas in this test setup
        with patch("documents.dev_api.schema_context"):
            response = view(request, tenant_id=tenant_id, document_id=doc_id)

    assert response.status_code == 200
    assert response.data["status"] == "deleted"
    assert response.data["mode"] == "hard"

    # Verify vector cleanup was called
    mock_vector_client.hard_delete_documents.assert_called_once()
    call_args = mock_vector_client.hard_delete_documents.call_args
    assert call_args.kwargs["tenant_id"] == str(
        mock_tenant_context.resolve_identifier.return_value.id
    )
    assert call_args.kwargs["document_ids"] == [doc.id]

    # Verify document.delete() was called
    doc.delete.assert_called_once()


@pytest.mark.django_db
def test_delete_document_soft_cleanup(
    api_factory, mock_vector_client, mock_tenant_context
):
    """Test that soft delete (default) triggers vector lifecycle update."""
    # Enable DEBUG to allow dev API usage
    settings.DEBUG = True

    # Create a mock document
    doc = Mock()
    doc.id = uuid4()
    doc_id = str(doc.id)
    doc.save = Mock()  # Soft delete calls save()
    doc.lifecycle_state = "active"

    tenant_id = "test-tenant"

    view = DocumentDevViewSet.as_view({"delete": "delete_document"})
    # No hard=true param
    request = api_factory.delete(f"/dev/{tenant_id}/{doc_id}/delete")

    # Mock database object retrieval to return our mock doc
    with patch("documents.dev_api.get_object_or_404", return_value=doc):
        # Mock schema_context since we don't have real tenant schemas in this test setup
        with patch("documents.dev_api.schema_context"):
            response = view(request, tenant_id=tenant_id, document_id=doc_id)

    assert response.status_code == 200
    assert response.data["status"] == "retired"
    assert response.data["mode"] == "soft"

    # Verify vector lifecycle update was called
    mock_vector_client.update_lifecycle_state.assert_called_once()
    call_args = mock_vector_client.update_lifecycle_state.call_args
    assert call_args.kwargs["tenant_id"] == str(
        mock_tenant_context.resolve_identifier.return_value.id
    )
    assert call_args.kwargs["document_ids"] == [doc.id]
    assert call_args.kwargs["state"] == "retired"
    assert call_args.kwargs["reason"] == "soft_delete_dev_api"

    # Verify document.save() was called (persisting retrieval)
    doc.save.assert_called_once()
