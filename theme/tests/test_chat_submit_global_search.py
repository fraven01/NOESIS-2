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

    captured: dict[str, object] = {}

    captured["meta"] = None  # Reset

    def fake_submit(graph_name, tool_context, state, **kwargs):
        captured["meta"] = {
            "tool_context": tool_context.model_dump(mode="json"),
            "graph_name": graph_name,
        }
        return {
            "status": "success",
            "data": {"result": {"answer": "ok", "snippets": []}},
        }, True

    with patch(
        "theme.helpers.tasks.submit_business_graph",
        side_effect=fake_submit,
    ):
        response = chat_submit(request)

    assert response.status_code == 200
    assert "ok" in response.content.decode()

    meta = captured["meta"]
    assert isinstance(meta, dict)
    tool_ctx = meta.get("tool_context")
    assert isinstance(tool_ctx, dict)
    # Check case_id in business part of tool_context
    assert tool_ctx.get("business", {}).get("case_id") is None
