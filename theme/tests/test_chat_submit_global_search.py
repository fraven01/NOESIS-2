from unittest.mock import patch

import pytest
from django.test import RequestFactory
from django.urls import reverse

from theme.views import chat_submit


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_chat_submit_global_search_does_not_require_case_id(tenant_pool):
    tenant = tenant_pool["alpha"]
    factory = RequestFactory()
    request = factory.post(
        reverse("chat-submit"),
        data={"message": "hello", "global_search": "on"},
    )
    request.tenant = tenant
    from django.contrib.sessions.backends.db import SessionStore

    request.session = SessionStore()

    with patch("theme.views_chat.RagQueryService.execute") as mock_execute:
        mock_execute.return_value = (
            {},
            {"answer": "ok", "snippets": []},
        )
        response = chat_submit(request)

    assert response.status_code == 200
    assert "ok" in response.content.decode()

    tool_context = mock_execute.call_args[1]["tool_context"]
    assert tool_context.business.case_id is None


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_chat_submit_case_scope_uses_case_id(tenant_pool):
    tenant = tenant_pool["alpha"]
    factory = RequestFactory()
    request = factory.post(
        reverse("chat-submit"),
        data={"message": "hello", "case_id": "case-scope", "chat_scope": "case"},
    )
    request.tenant = tenant
    from django.contrib.sessions.backends.db import SessionStore

    request.session = SessionStore()

    with patch("theme.views_chat.RagQueryService.execute") as mock_execute:
        mock_execute.return_value = (
            {},
            {"answer": "ok", "snippets": []},
        )
        response = chat_submit(request)

    assert response.status_code == 200
    assert "ok" in response.content.decode()

    tool_context = mock_execute.call_args[1]["tool_context"]
    assert tool_context.business.case_id == "case-scope"
