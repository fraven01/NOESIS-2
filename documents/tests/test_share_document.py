from __future__ import annotations

import json

import pytest
from django.test import RequestFactory

from customers.models import Tenant
from documents.models import Document, DocumentPermission
from documents.views import share_document
from profiles.models import UserProfile
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def _create_document(*, tenant: Tenant, created_by) -> Document:
    return Document.objects.create(
        tenant=tenant,
        hash="share-hash",
        source="upload",
        metadata={},
        created_by=created_by,
    )


def test_share_document_owner_grants_permission(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    owner = UserFactory()
    target_user = UserFactory()
    document = _create_document(tenant=tenant, created_by=owner)

    factory = RequestFactory()
    request = factory.post(
        f"/documents/share/{document.id}/",
        data=json.dumps(
            {
                "user_id": str(target_user.id),
                "permission_type": DocumentPermission.PermissionType.VIEW,
            }
        ),
        content_type="application/json",
    )
    request.user = owner
    request.tenant = tenant

    response = share_document(request, str(document.id))

    assert response.status_code == 201
    assert DocumentPermission.objects.filter(
        document=document,
        user=target_user,
        permission_type=DocumentPermission.PermissionType.VIEW,
    ).exists()


def test_share_document_forbidden_for_non_owner(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    owner = UserFactory()
    outsider = UserFactory()
    target_user = UserFactory()
    document = _create_document(tenant=tenant, created_by=owner)

    factory = RequestFactory()
    request = factory.post(
        f"/documents/share/{document.id}/",
        data=json.dumps({"user_id": str(target_user.id)}),
        content_type="application/json",
    )
    request.user = outsider
    request.tenant = tenant

    response = share_document(request, str(document.id))

    assert response.status_code == 403
    assert not DocumentPermission.objects.filter(
        document=document,
        user=target_user,
    ).exists()


def test_share_document_allows_tenant_admin(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    owner = UserFactory()
    admin = UserFactory(role=UserProfile.Roles.TENANT_ADMIN)
    target_user = UserFactory()
    document = _create_document(tenant=tenant, created_by=owner)

    factory = RequestFactory()
    request = factory.post(
        f"/documents/share/{document.id}/",
        data=json.dumps(
            {"user_id": str(target_user.id), "permission_type": "DOWNLOAD"}
        ),
        content_type="application/json",
    )
    request.user = admin
    request.tenant = tenant

    response = share_document(request, str(document.id))

    assert response.status_code == 201
    assert DocumentPermission.objects.filter(
        document=document,
        user=target_user,
        permission_type=DocumentPermission.PermissionType.DOWNLOAD,
    ).exists()
