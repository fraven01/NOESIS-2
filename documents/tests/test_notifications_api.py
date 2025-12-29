from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from customers.models import Tenant
from documents.models import Document, DocumentNotification
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def test_notifications_mark_read(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory()

    document = Document.objects.create(
        tenant=tenant,
        hash="notify-hash",
        source="upload",
        metadata={},
        created_by=user,
    )

    notification = DocumentNotification.objects.create(
        user=user,
        document=document,
        event_type=DocumentNotification.EventType.COMMENT,
        payload={"note": "test"},
    )

    client = APIClient()
    client.force_authenticate(user=user)

    response = client.patch(
        f"/documents/api/notifications/{notification.id}/",
        {},
        format="json",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert response.status_code == 200
    notification.refresh_from_db()
    assert notification.read_at is not None


def test_notifications_mark_all_read(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory()

    document = Document.objects.create(
        tenant=tenant,
        hash="notify-hash-2",
        source="upload",
        metadata={},
        created_by=user,
    )

    DocumentNotification.objects.create(
        user=user,
        document=document,
        event_type=DocumentNotification.EventType.COMMENT,
    )

    client = APIClient()
    client.force_authenticate(user=user)

    response = client.post(
        "/documents/api/notifications/mark_all_read/",
        {},
        format="json",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert response.status_code == 200
    assert (
        DocumentNotification.objects.filter(user=user, read_at__isnull=True).count()
        == 0
    )
