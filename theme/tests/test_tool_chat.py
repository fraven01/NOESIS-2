import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_tool_chat_allows_global_scope_without_case(auth_client):
    session = auth_client.session
    session["rag_chat_scope"] = "global"
    session.pop("rag_active_case_id", None)
    session.save()

    response = auth_client.get(reverse("tool-chat"))

    assert response.status_code == 200
    assert response.context["chat_scope"] == "global"
    assert response.context["case_id"] is None
