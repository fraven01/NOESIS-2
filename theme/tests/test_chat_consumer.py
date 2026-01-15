from __future__ import annotations

from asgiref.sync import async_to_sync
from unittest.mock import AsyncMock

import pytest

from theme.consumers import CHECKPOINTER, RagChatConsumer, RagQueryService


@pytest.mark.django_db
def test_rag_chat_consumer_uses_service(monkeypatch, user, tenant_pool):
    tenant = tenant_pool["alpha"]
    consumer = RagChatConsumer()
    consumer.scope = {"user": user}
    consumer.channel_name = "test"
    consumer.send_json = AsyncMock()

    monkeypatch.setattr(
        CHECKPOINTER,
        "load",
        lambda ctx: {"chat_history": []},
    )
    monkeypatch.setattr(
        CHECKPOINTER,
        "save",
        lambda ctx, state: None,
    )

    captured: dict[str, object] = {}

    def fake_execute(
        self,
        *,
        tool_context,
        question,
        hybrid=None,
        chat_history=None,
        graph_state=None,
    ):
        captured["tool_context"] = tool_context
        captured["chat_history"] = list(chat_history or [])
        captured["question"] = question
        captured["graph_state"] = graph_state
        return (
            {},
            {
                "answer": "ok",
                "prompt_version": "v1",
                "retrieval": {
                    "alpha": 0.5,
                    "min_sim": 0.1,
                    "top_k_effective": 1,
                    "matches_returned": 0,
                    "max_candidates_effective": 1,
                    "vector_candidates": 0,
                    "lexical_candidates": 0,
                    "deleted_matches_blocked": 0,
                    "visibility_effective": "active",
                    "took_ms": 1,
                    "routing": {
                        "profile": "standard",
                        "vector_space_id": "rag/standard@v1",
                    },
                },
                "snippets": [],
            },
        )

    monkeypatch.setattr(RagQueryService, "execute", fake_execute)

    payload = {
        "message": "hello",
        "tenant_id": str(tenant.id),
        "tenant_schema": tenant.schema_name,
        "case_id": "chat-case",
    }

    async_to_sync(consumer.receive_json)(payload)

    assert captured["question"] == "hello"
    tool_context = captured["tool_context"]
    assert tool_context.business.case_id == "chat-case"
    assert captured["chat_history"] == []
    assert consumer.send_json.call_args_list[-1][0][0]["event"] == "final"
