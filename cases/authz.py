"""Case-level authorization helpers (TENANT-SCOPED).

IMPORTANT: Authorization vs. Account Policies
==============================================
This module handles TENANT-SPECIFIC authorization:
- User roles may differ per tenant
- Access rules depend on tenant type (ENTERPRISE vs LAW_FIRM)
- Case membership is tenant-scoped
- Works_council_scope is per-tenant policy

For USER-GLOBAL account policies (expiry, active status), see:
- profiles/policies.py - Account policy enforcement
- profiles/authentication.py - Authentication-time policy checks

Execution Order:
1. Account Policies (profiles/policies.py) - Authentication time, user-global
2. Authorization (this module) - View/object level, tenant-specific
"""

from __future__ import annotations

from django.contrib.auth.models import User

from cases.models import Case, CaseMembership
from customers.models import Tenant
from profiles.models import UserProfile


def user_can_access_case(user: User, case: Case, tenant: Tenant | None = None) -> bool:
    """Return whether user can access the given case."""

    if not user or not user.is_authenticated:
        return False

    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return False

    if not profile.is_active:
        return False

    # EXTERNAL accounts are ALWAYS case-scoped, regardless of role
    if profile.account_type == UserProfile.AccountType.EXTERNAL:
        return CaseMembership.objects.filter(case=case, user=user).exists()

    # Determine tenant type
    if tenant is None:
        tenant = case.tenant

    tenant_type = getattr(tenant, "tenant_type", Tenant.TenantType.ENTERPRISE)

    # ENTERPRISE scope rules
    if tenant_type == Tenant.TenantType.ENTERPRISE:
        # All-cases access for certain roles
        if profile.role in (
            UserProfile.Roles.TENANT_ADMIN,
            UserProfile.Roles.LEGAL,
            UserProfile.Roles.MANAGEMENT,
        ):
            return True

        # WORKS_COUNCIL: depends on tenant policy
        if profile.role == UserProfile.Roles.WORKS_COUNCIL:
            scope = getattr(tenant, "works_council_scope", Tenant.WorksCouncilScope.ALL)
            if scope == Tenant.WorksCouncilScope.ALL:
                return True
            # else fall through to membership check

        # STAKEHOLDER or WORKS_COUNCIL with ASSIGNED scope
        return CaseMembership.objects.filter(case=case, user=user).exists()

    # LAW_FIRM scope rules (default: assigned only)
    if tenant_type == Tenant.TenantType.LAW_FIRM:
        # Only TENANT_ADMIN has all-cases access
        if profile.role == UserProfile.Roles.TENANT_ADMIN:
            return True

        # Everyone else (including LEGAL, MANAGEMENT) is case-scoped
        return CaseMembership.objects.filter(case=case, user=user).exists()

    # Unknown tenant type: deny
    return False


def get_accessible_cases_queryset(user: User, tenant: Tenant):
    """Return queryset of cases accessible to user in tenant."""

    if not user or not user.is_authenticated:
        return Case.objects.none()

    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return Case.objects.none()

    if not profile.is_active:
        return Case.objects.none()

    base_qs = Case.objects.filter(tenant=tenant)

    # EXTERNAL accounts: always case-scoped
    if profile.account_type == UserProfile.AccountType.EXTERNAL:
        # Use JOIN instead of subquery for better performance
        return base_qs.filter(memberships__user=user).distinct()

    tenant_type = getattr(tenant, "tenant_type", Tenant.TenantType.ENTERPRISE)

    # ENTERPRISE all-access roles
    if tenant_type == Tenant.TenantType.ENTERPRISE:
        # All-cases access for certain roles
        if profile.role in (
            UserProfile.Roles.TENANT_ADMIN,
            UserProfile.Roles.LEGAL,
            UserProfile.Roles.MANAGEMENT,
        ):
            return base_qs

        # WORKS_COUNCIL: depends on tenant policy
        if profile.role == UserProfile.Roles.WORKS_COUNCIL:
            scope = getattr(tenant, "works_council_scope", Tenant.WorksCouncilScope.ALL)
            if scope == Tenant.WorksCouncilScope.ALL:
                return base_qs

        # STAKEHOLDER or scoped WORKS_COUNCIL: membership-based
        return base_qs.filter(memberships__user=user).distinct()

    # LAW_FIRM: assigned-only by default
    if tenant_type == Tenant.TenantType.LAW_FIRM:
        # Only TENANT_ADMIN has all-cases access
        if profile.role == UserProfile.Roles.TENANT_ADMIN:
            return base_qs

        # Everyone else is case-scoped
        return base_qs.filter(memberships__user=user).distinct()

    # Unknown tenant type: deny all
    return Case.objects.none()
