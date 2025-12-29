from __future__ import annotations

from datetime import timedelta

import pytest
from django.utils import timezone

from cases.models import Case, CaseMembership
from customers.models import Tenant
from documents.authz import DocumentAuthzService
from documents.models import Document, DocumentPermission
from profiles.models import UserProfile
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def _create_document(
    *, tenant: Tenant, case_id: str | None = None, created_by=None
) -> Document:
    return Document.objects.create(
        tenant=tenant,
        hash="authz-hash",
        source="upload",
        metadata={},
        case_id=case_id,
        created_by=created_by,
    )


def test_explicit_permission_grants_access(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)
    document = _create_document(tenant=tenant)

    DocumentPermission.objects.create(
        document=document,
        user=user,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    access = DocumentAuthzService.user_can_access_document(
        user=user,
        document=document,
        permission_type=DocumentPermission.PermissionType.VIEW,
        tenant=tenant,
    )

    assert access.allowed is True
    assert access.source == "document_permission"


def test_expired_permission_denies_access(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)
    document = _create_document(tenant=tenant)

    DocumentPermission.objects.create(
        document=document,
        user=user,
        permission_type=DocumentPermission.PermissionType.VIEW,
        expires_at=timezone.now() - timedelta(hours=1),
    )

    access = DocumentAuthzService.user_can_access_document(
        user=user,
        document=document,
        permission_type=DocumentPermission.PermissionType.VIEW,
        tenant=tenant,
    )

    assert access.allowed is False
    assert access.reason == "no_permission"


def test_case_membership_grants_access(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)

    case = Case.objects.create(tenant=tenant, external_id="CASE-101")
    CaseMembership.objects.create(case=case, user=user)

    document = _create_document(tenant=tenant, case_id="CASE-101")

    access = DocumentAuthzService.user_can_access_document(
        user=user,
        document=document,
        permission_type=DocumentPermission.PermissionType.VIEW,
        tenant=tenant,
    )

    assert access.allowed is True
    assert access.source == "case_membership"


def test_owner_grants_access(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    owner = UserFactory()
    document = _create_document(tenant=tenant, created_by=owner)

    access = DocumentAuthzService.user_can_access_document(
        user=owner,
        document=document,
        permission_type=DocumentPermission.PermissionType.VIEW,
        tenant=tenant,
    )

    assert access.allowed is True
    assert access.source == "owner"


def test_tenant_admin_grants_access(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    admin = UserFactory(role=UserProfile.Roles.TENANT_ADMIN)
    document = _create_document(tenant=tenant)

    access = DocumentAuthzService.user_can_access_document(
        user=admin,
        document=document,
        permission_type=DocumentPermission.PermissionType.VIEW,
        tenant=tenant,
    )

    assert access.allowed is True
    assert access.source == "tenant_role"
