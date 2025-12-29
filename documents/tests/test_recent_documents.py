from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory
from django.utils import timezone

from customers.models import Tenant
from documents.models import Document, DocumentActivity
from documents.views import recent_documents
from users.models import User


pytestmark = pytest.mark.django_db


def _create_document(*, tenant: Tenant, source: str) -> Document:
    return Document.objects.create(
        tenant=tenant,
        hash=f"{source}-hash",
        source=source,
        metadata={},
    )


def test_recent_documents_requires_authentication(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    factory = RequestFactory()
    request = factory.get("/documents/recent/")
    request.user = AnonymousUser()
    request.tenant = SimpleNamespace(
        tenant_id=tenant.schema_name, schema_name=tenant.schema_name
    )

    response = recent_documents(request)

    assert response.status_code == 401


def test_recent_documents_returns_latest_unique(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = User.objects.create_user(username="recent-user")

    doc_a = _create_document(tenant=tenant, source="upload-a")
    doc_b = _create_document(tenant=tenant, source="upload-b")

    now = timezone.now()
    activity_a_old = DocumentActivity.objects.create(
        document=doc_a,
        user=user,
        activity_type=DocumentActivity.ActivityType.DOWNLOAD,
    )
    DocumentActivity.objects.filter(id=activity_a_old.id).update(
        timestamp=now - timedelta(minutes=10)
    )

    activity_b = DocumentActivity.objects.create(
        document=doc_b,
        user=user,
        activity_type=DocumentActivity.ActivityType.VIEW,
    )
    DocumentActivity.objects.filter(id=activity_b.id).update(
        timestamp=now - timedelta(minutes=5)
    )

    activity_a_new = DocumentActivity.objects.create(
        document=doc_a,
        user=user,
        activity_type=DocumentActivity.ActivityType.DOWNLOAD,
    )
    DocumentActivity.objects.filter(id=activity_a_new.id).update(timestamp=now)

    factory = RequestFactory()
    request = factory.get("/documents/recent/")
    request.user = user
    request.tenant = SimpleNamespace(
        tenant_id=tenant.schema_name, schema_name=tenant.schema_name
    )

    response = recent_documents(request)

    assert response.status_code == 200
    payload = json.loads(response.content)
    assert [item["document_id"] for item in payload] == [
        str(doc_a.id),
        str(doc_b.id),
    ]
