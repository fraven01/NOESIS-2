import pytest
import uuid
from django.urls import reverse

from cases.tests.factories import CaseFactory
from customers.models import Tenant
from documents.models import DocumentCollection


@pytest.mark.django_db
def test_tool_chat_context_includes_collections(auth_client, test_tenant_schema_name):
    """Verify tool_chat view provides collection_options and active_collection context."""
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    case = CaseFactory(title="Chat Case", tenant=tenant)
    col1 = DocumentCollection.objects.create(
        name="Chat Collection",
        case=case,
        tenant=tenant,
        collection_id=uuid.uuid4(),
        key=f"chat-col-{uuid.uuid4()}",
    )

    session = auth_client.session
    session["rag_active_collection_id"] = str(col1.collection_id)
    session["rag_active_case_id"] = str(case.id)
    session.save()

    url = reverse("tool-chat")
    response = auth_client.get(url)
    assert response.status_code == 200

    assert "collection_options" in response.context
    assert response.context["chat_scope"] == "collection"
    assert any(
        o["id"] == str(col1.collection_id)
        for o in response.context["collection_options"]
    )
    assert response.context["active_collection_id"] == str(col1.collection_id)
    assert response.context["case_id"] == str(case.id)

    content = response.content.decode()
    assert "Chat Collection" in content
    assert f'value="{str(case.id)}"' in content
    assert 'name="chat_scope"' in content


@pytest.mark.django_db
def test_tool_search_context_includes_collections(auth_client, test_tenant_schema_name):
    """Verify tool_search view provides collection_options and active_collection context."""
    # Setup
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    case = CaseFactory(title="Test Case", tenant=tenant)

    col1 = DocumentCollection.objects.create(
        name="Collection A",
        case=case,
        tenant=tenant,
        collection_id=uuid.uuid4(),
        key=f"col-a-{uuid.uuid4()}",
    )
    col2 = DocumentCollection.objects.create(
        name="Collection B",
        case=case,
        tenant=tenant,
        collection_id=uuid.uuid4(),
        key=f"col-b-{uuid.uuid4()}",
    )

    # Set session context
    session = auth_client.session
    session["rag_active_collection_id"] = str(col1.collection_id)
    session.save()

    url = reverse("tool-search")
    response = auth_client.get(url)

    assert response.status_code == 200

    # Check context
    assert "collection_options" in response.context
    opts = response.context["collection_options"]
    assert len(opts) >= 2
    assert any(o["id"] == str(col1.collection_id) for o in opts)
    assert any(o["id"] == str(col2.collection_id) for o in opts)

    assert response.context["active_collection_id"] == str(col1.collection_id)

    # Check HTML content
    content = response.content.decode()
    assert "Collection A" in content
    assert "Target Collection" in content
    assert "(Ingestion)" in content
