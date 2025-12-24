from unittest.mock import patch

import pytest
from django.test import RequestFactory
from django.urls import reverse

from customers.tests.factories import TenantFactory
from theme.views import chat_submit


@pytest.mark.django_db
def test_chat_submit_global_search_does_not_require_case_id():
    tenant = TenantFactory(schema_name="workbench")
    factory = RequestFactory()

    request = factory.post(
        reverse("chat-submit"),
        data={"message": "hello", "global_search": "on"},
    )
    request.tenant = tenant

    captured: dict[str, object] = {}

    def fake_run(state, meta):
        import importlib

        rag = importlib.import_module(
            "ai_core.graphs.technical.retrieval_augmented_generation"
        )
        rag._build_tool_context(meta)
        captured["meta"] = meta
        return state, {"answer": "ok", "snippets": []}

    with patch(
        "ai_core.graphs.technical.retrieval_augmented_generation.run",
        side_effect=fake_run,
    ):
        response = chat_submit(request)

    assert response.status_code == 200
    assert "ok" in response.content.decode()

    meta = captured["meta"]
    assert isinstance(meta, dict)
    assert meta.get("scope_context", {}).get("case_id") is None
    assert isinstance(meta.get("tool_context"), dict)
