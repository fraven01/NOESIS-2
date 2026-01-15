"""Document authorization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from django.db.models import Q
from django.utils import timezone

from cases.authz import get_accessible_cases_queryset, user_can_access_case
from cases.models import Case
from customers.models import Tenant
from profiles.models import UserProfile

from .models import Document, DocumentPermission


@dataclass(frozen=True)
class DocumentAccessResult:
    allowed: bool
    source: str | None = None
    reason: str | None = None


def _user_has_tenant_wide_access(profile: UserProfile, tenant: Tenant | None) -> bool:
    if profile.account_type == UserProfile.AccountType.EXTERNAL:
        return False

    tenant_type = getattr(tenant, "tenant_type", Tenant.TenantType.ENTERPRISE)

    if tenant_type == Tenant.TenantType.ENTERPRISE:
        if profile.role in (
            UserProfile.Roles.TENANT_ADMIN,
            UserProfile.Roles.LEGAL,
            UserProfile.Roles.MANAGEMENT,
        ):
            return True
        if profile.role == UserProfile.Roles.WORKS_COUNCIL:
            scope = getattr(tenant, "works_council_scope", Tenant.WorksCouncilScope.ALL)
            return scope == Tenant.WorksCouncilScope.ALL
        return False

    if tenant_type == Tenant.TenantType.LAW_FIRM:
        return profile.role == UserProfile.Roles.TENANT_ADMIN

    return False


def user_has_tenant_wide_access(*, user, tenant: Tenant | None) -> bool:
    if not user or not getattr(user, "is_authenticated", False):
        return False
    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return False
    if not profile.is_active:
        return False
    return _user_has_tenant_wide_access(profile, tenant)


class DocumentAuthzService:
    """Document-level authorization."""

    @staticmethod
    def user_can_access_document(
        *,
        user,
        document: Document,
        permission_type: str = DocumentPermission.PermissionType.VIEW,
        tenant: Tenant | None = None,
    ) -> DocumentAccessResult:
        if not user or not getattr(user, "is_authenticated", False):
            return DocumentAccessResult(False, reason="unauthenticated")

        try:
            profile = user.userprofile
        except UserProfile.DoesNotExist:
            return DocumentAccessResult(False, reason="profile_missing")

        if not profile.is_active:
            return DocumentAccessResult(False, reason="inactive")

        if document.created_by_id and document.created_by_id == user.id:
            return DocumentAccessResult(True, source="owner")

        now = timezone.now()
        has_permission = DocumentPermission.objects.filter(
            document=document,
            user=user,
            permission_type=permission_type,
        ).filter(Q(expires_at__isnull=True) | Q(expires_at__gt=now))

        if has_permission.exists():
            return DocumentAccessResult(True, source="document_permission")

        tenant = tenant or document.tenant

        if document.case_id:
            case = Case.objects.filter(
                tenant=tenant, external_id=document.case_id
            ).first()
            if case and user_can_access_case(user, case, tenant):
                return DocumentAccessResult(True, source="case_membership")

        if tenant and _user_has_tenant_wide_access(profile, tenant):
            return DocumentAccessResult(True, source="tenant_role")

        return DocumentAccessResult(False, reason="no_permission")

    @staticmethod
    def user_can_access_document_id(
        *,
        user,
        document_id,
        permission_type: str = DocumentPermission.PermissionType.VIEW,
        tenant: Tenant | None = None,
    ) -> DocumentAccessResult:
        document = (
            Document.objects.filter(id=document_id).select_related("tenant").first()
        )
        if document is None:
            return DocumentAccessResult(False, reason="not_found")
        return DocumentAuthzService.user_can_access_document(
            user=user,
            document=document,
            permission_type=permission_type,
            tenant=tenant,
        )

    @staticmethod
    def accessible_documents_queryset(
        *,
        user,
        tenant: Tenant,
        permission_type: str = DocumentPermission.PermissionType.VIEW,
    ):
        """Return queryset of documents accessible to a user within a tenant."""

        if not user or not getattr(user, "is_authenticated", False):
            return Document.objects.none()

        try:
            profile = user.userprofile
        except UserProfile.DoesNotExist:
            return Document.objects.none()

        if not profile.is_active:
            return Document.objects.none()

        base_qs = Document.objects.filter(tenant=tenant)

        if _user_has_tenant_wide_access(profile, tenant):
            return base_qs

        now = timezone.now()
        permission_doc_ids = DocumentPermission.objects.filter(
            user=user,
            permission_type=permission_type,
        ).filter(Q(expires_at__isnull=True) | Q(expires_at__gt=now))

        accessible_case_ids = get_accessible_cases_queryset(user, tenant).values(
            "external_id"
        )

        return base_qs.filter(
            Q(created_by_id=user.id)
            | Q(id__in=permission_doc_ids.values("document_id"))
            | Q(case_id__in=accessible_case_ids)
        )
