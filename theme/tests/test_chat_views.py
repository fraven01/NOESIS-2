from unittest.mock import patch

import pytest
from django.test import RequestFactory
from django.urls import reverse

from theme.views import chat_submit


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_htmx_chat_uses_service(tenant_pool):
    tenant = tenant_pool["alpha"]
    factory = RequestFactory()
    request = factory.post(
        reverse("chat-submit"),
        data={"message": "hello"},
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
    assert mock_execute.called
