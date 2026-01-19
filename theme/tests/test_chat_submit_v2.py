from __future__ import annotations

from unittest.mock import patch

import pytest
from django.test import RequestFactory
from django.urls import reverse

from theme.views import chat_submit


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_chat_submit_v2_limits_sources_and_links_citations(
    tenant_pool, settings, monkeypatch
):
    settings.DEBUG = True
    tenant = tenant_pool["alpha"]
    factory = RequestFactory()
    request = factory.post(
        reverse("chat-submit"),
        data={"message": "hello"},
    )
    request.tenant = tenant
    from django.contrib.sessions.backends.db import SessionStore

    request.session = SessionStore()

    monkeypatch.setattr(
        "theme.chat_utils._build_download_url",
        lambda doc_id: f"/download/{doc_id}",
    )

    used_sources = [
        {"id": "s1", "label": "Doc 1", "relevance_score": 0.9},
        {"id": "s2", "label": "Doc 2", "relevance_score": 0.8},
        {"id": "s3", "label": "Doc 3", "relevance_score": 0.7},
        {"id": "s4", "label": "Doc 4", "relevance_score": 0.6},
    ]
    snippets = [
        {"document_id": "d1", "text": "t1", "source": "Doc 1", "score": 0.9},
        {"document_id": "d2", "text": "t2", "source": "Doc 2", "score": 0.8},
        {"document_id": "d3", "text": "t3", "source": "Doc 3", "score": 0.7},
        {"document_id": "d4", "text": "t4", "source": "Doc 4", "score": 0.6},
    ]

    with patch("theme.views_chat.RagQueryService.execute") as mock_execute:
        mock_execute.return_value = (
            {},
            {
                "answer": "Answer [Doc 1]",
                "snippets": snippets,
                "used_sources": used_sources,
                "retrieval": {"top_k_effective": 3},
            },
        )
        response = chat_submit(request)

    body = response.content.decode()
    assert response.status_code == 200
    assert "/download/d1" in body
    assert "Doc 1" in body
    assert "Doc 2" in body
    assert "Doc 3" in body
    assert "Doc 4" not in body
