import pytest
from django.urls import reverse


@pytest.mark.django_db
class TestWorkbenchContext:
    def test_set_context_updates_session(self, client, user, tenant):
        """Verify that the set_context endpoint updates the session correctly."""
        client.force_login(user)
        url = reverse("rag_tools_set_context")

        # Test setting Collection and Case
        response = client.post(
            url, {"collection_id": "test-collection", "case_id": "test-case"}
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
        response = client.post(url, {"collection_id": "", "case_id": ""})

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
