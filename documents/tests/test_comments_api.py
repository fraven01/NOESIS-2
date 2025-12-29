from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from customers.models import Tenant
from documents.models import (
    Document,
    DocumentMention,
    DocumentNotification,
    DocumentPermission,
)
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def test_comment_creates_mentions_and_notifications(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    author = UserFactory()
    mentioned = UserFactory(username="mentioned-user")

    document = Document.objects.create(
        tenant=tenant,
        hash="comment-hash",
        source="upload",
        metadata={},
        created_by=author,
    )

    DocumentPermission.objects.create(
        document=document,
        user=mentioned,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    client = APIClient()
    client.force_authenticate(user=author)

    response = client.post(
        "/documents/api/comments/",
        {"document": str(document.id), "text": f"Hello <@{mentioned.id}>"},
        format="json",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert response.status_code == 201
    comment_id = response.data["id"]

    assert DocumentMention.objects.filter(
        comment_id=comment_id, mentioned_user=mentioned
    ).exists()
    assert DocumentNotification.objects.filter(
        user=mentioned, event_type=DocumentNotification.EventType.MENTION
    ).exists()

    list_response = client.get(
        f"/documents/api/comments/?document_id={document.id}",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert list_response.status_code == 200
    assert len(list_response.data) == 1
