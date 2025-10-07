from types import SimpleNamespace
import uuid
from types import SimpleNamespace
import uuid

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIRequestFactory

from ai_core.views import RagHardDeleteAdminView, _resolve_hard_delete_actor
from profiles.models import UserProfile


@pytest.mark.django_db
def test_rag_hard_delete_admin_service_key(client, settings, monkeypatch):
    settings.RAG_INTERNAL_KEYS = ["ops-service"]
    tenant_id = str(uuid.uuid4())
    captured: dict[str, object] = {}

    def fake_delay(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(id="job-123")

    monkeypatch.setattr(
        "ai_core.views.hard_delete", SimpleNamespace(delay=fake_delay)
    )

    response = client.post(
        "/ai/rag/admin/hard-delete/",
        data={
            "tenant_id": tenant_id,
            "document_ids": [
                "3fbb07d0-2a5b-4b75-8ad4-5c5e8f3e1d21",
                "986cf6d5-2d8c-4b6c-98eb-3ac80f8aa84f",
            ],
            "reason": "cleanup",
            "ticket_ref": "TCK-1234",
        },
        content_type="application/json",
        HTTP_X_INTERNAL_KEY="ops-service",
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"] == "job-123"
    assert body["trace_id"]
    assert body["documents_requested"] == 2
    assert response["X-Trace-Id"]

    assert captured["args"] == (
        tenant_id,
        [
            "3fbb07d0-2a5b-4b75-8ad4-5c5e8f3e1d21",
            "986cf6d5-2d8c-4b6c-98eb-3ac80f8aa84f",
        ],
        "cleanup",
        "TCK-1234",
    )
    assert captured["kwargs"]["actor"] == {"internal_key": "ops-service"}
    assert captured["kwargs"]["trace_id"] == body["trace_id"]


@pytest.mark.django_db
def test_rag_hard_delete_admin_requires_authorisation(client, settings):
    settings.RAG_INTERNAL_KEYS = ["ops-service"]
    response = client.post(
        "/ai/rag/admin/hard-delete/",
        data={
            "tenant_id": str(uuid.uuid4()),
            "document_ids": [str(uuid.uuid4())],
            "reason": "cleanup",
            "ticket_ref": "TCK-9999",
        },
        content_type="application/json",
    )

    assert response.status_code == 403


@pytest.mark.django_db
def test_rag_hard_delete_admin_rejects_non_admin_user(client):
    user_model = get_user_model()
    user = user_model.objects.create_user("guest", "guest@example.com", "pass")
    UserProfile.objects.update_or_create(user=user, defaults={"role": UserProfile.Roles.GUEST})
    client.force_login(user)
    client.cookies["csrftoken"] = "test-token"

    response = client.post(
        "/ai/rag/admin/hard-delete/",
        data={
            "tenant_id": str(uuid.uuid4()),
            "document_ids": [str(uuid.uuid4())],
            "reason": "cleanup",
            "ticket_ref": "TCK-9999",
        },
        content_type="application/json",
        HTTP_X_CSRFTOKEN="test-token",
    )

    assert response.status_code == 403


@pytest.mark.django_db
def test_resolve_hard_delete_actor_allows_admin_user(monkeypatch):
    user_model = get_user_model()
    user = user_model.objects.create_user("admin", "admin@example.com", "pass")
    UserProfile.objects.update_or_create(
        user=user,
        defaults={"role": UserProfile.Roles.ADMIN, "is_active": True},
    )

    request = SimpleNamespace(headers={}, META={}, user=user, session={})

    monkeypatch.setattr(
        "ai_core.views.allow_extended_visibility", lambda _: True
    )

    actor = _resolve_hard_delete_actor(request, "Alex Admin")

    assert actor == {"user_id": user.pk, "label": "Alex Admin"}
