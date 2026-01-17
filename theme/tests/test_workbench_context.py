from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.urls import reverse

from ai_core.tool_contracts import ContextError
from theme.helpers.context import prepare_workbench_context


@pytest.mark.django_db
class TestWorkbenchContext:
    def test_set_context_updates_session(self, client, user, tenant):
        """Verify that the set_context endpoint updates the session correctly."""
        client.force_login(user)
        url = reverse("rag_tools_set_context")

        # Test setting Collection and Case
        response = client.post(
            url,
            {
                "collection_id": "test-collection",
                "case_id": "test-case",
                "workflow_id": "workflow-test",
            },
        )
        assert response.status_code == 200
        assert client.session.get("rag_active_collection_id") == "test-collection"
        assert client.session.get("rag_active_case_id") == "test-case"

    def test_set_context_clears_values(self, client, user, tenant):
        """Verify that sending empty values clears the session state (Global mode)."""
        client.force_login(user)

        # Pre-set session
        session = client.session
        session["rag_active_collection_id"] = "old-col"
        session["rag_active_case_id"] = "old-case"
        session.save()

        url = reverse("rag_tools_set_context")
        response = client.post(
            url, {"collection_id": "", "case_id": "", "workflow_id": "workflow-test"}
        )

        assert response.status_code == 200
        assert client.session.get("rag_active_collection_id") is None
        assert client.session.get("rag_active_case_id") is None

    def test_workbench_index_context_loading(self, client, user, tenant):
        """Verify that the workbench index loads available options."""
        client.force_login(user)
        url = reverse("rag-tools")
        response = client.get(url)

        assert response.status_code == 200
        content = response.content.decode()
        # Workbench template should render its action headers
        assert "Active Case" in content
        assert "Active Collection" in content

    def test_prepare_workbench_context_with_asgi_scope(self):
        user_pk = uuid4()
        scope = {"user": SimpleNamespace(pk=user_pk, is_authenticated=True)}

        context = prepare_workbench_context(
            scope,
            tenant_id="tenant-1",
            tenant_schema="tenant-1",
            workflow_id="rag-chat-manual",
            thread_id="thread-1",
        )

        assert context.scope.tenant_id == "tenant-1"
        assert context.scope.user_id == str(user_pk)
        assert context.business.thread_id == "thread-1"

    def test_prepare_workbench_context_requires_tenant_id_for_asgi(self):
        scope = {"user": SimpleNamespace(pk=uuid4(), is_authenticated=True)}

        with pytest.raises(ContextError):
            prepare_workbench_context(scope, workflow_id="rag-chat-manual")
