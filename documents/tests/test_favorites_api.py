from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from customers.models import Tenant
from documents.models import Document, UserDocumentFavorite
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def test_favorites_create_and_list(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory()

    document = Document.objects.create(
        tenant=tenant,
        hash="fav-hash",
        source="upload",
        metadata={},
        created_by=user,
    )

    client = APIClient()
    client.force_authenticate(user=user)

    response = client.post(
        "/documents/api/favorites/",
        {"document": str(document.id)},
        format="json",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert response.status_code == 201
    assert UserDocumentFavorite.objects.filter(
        user=user, document=document
    ).exists()

    list_response = client.get(
        "/documents/api/favorites/",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert list_response.status_code == 200
    assert len(list_response.data) == 1
