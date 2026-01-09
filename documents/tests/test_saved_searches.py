from __future__ import annotations

import pytest
from django.utils import timezone
from rest_framework.test import APIClient

from customers.models import Tenant
from documents.models import Document, DocumentNotification, SavedSearch
from documents.tasks import run_saved_search_alerts
from users.tests.factories import UserFactory


pytestmark = [
    pytest.mark.slow,
    pytest.mark.django_db,
    pytest.mark.xdist_group("tenant_ops"),
]


def test_saved_search_scheduler_creates_notification(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory()

    Document.objects.create(
        tenant=tenant,
        hash="search-hash",
        source="upload",
        metadata={
            "normalized_document": {
                "meta": {"title": "Alpha Contract"},
            }
        },
        created_by=user,
    )

    saved_search = SavedSearch.objects.create(
        user=user,
        name="Alpha",
        query="Alpha",
        enable_alerts=True,
        next_run_at=timezone.now() - timezone.timedelta(minutes=5),
    )

    result = run_saved_search_alerts()

    assert result["processed"] >= 1
    assert DocumentNotification.objects.filter(
        user=user, event_type=DocumentNotification.EventType.SAVED_SEARCH
    ).exists()
    saved_search.refresh_from_db()
    assert saved_search.last_run_at is not None


def test_saved_search_api_create(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory()

    client = APIClient()
    client.force_authenticate(user=user)

    response = client.post(
        "/documents/api/saved-searches/",
        {"name": "My Search", "query": "contract"},
        format="json",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert response.status_code == 201
    assert SavedSearch.objects.filter(user=user, name="My Search").exists()
