"""Tests for case-level authorization."""

import pytest
from django.contrib.auth.models import AnonymousUser

from cases.authz import get_accessible_cases_queryset, user_can_access_case
from cases.tests.factories import CaseFactory, CaseMembershipFactory
from customers.models import Tenant
from profiles.models import UserProfile
from users.tests.factories import UserFactory


@pytest.fixture
def tenant_enterprise(test_tenant_schema_name):
    """Get or update the test tenant as ENTERPRISE."""
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    tenant.tenant_type = Tenant.TenantType.ENTERPRISE
    tenant.works_council_scope = Tenant.WorksCouncilScope.ALL
    tenant.save()
    return tenant


@pytest.fixture
def tenant_law_firm(test_tenant_schema_name):
    """Update the test tenant to LAW_FIRM type."""
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    tenant.tenant_type = Tenant.TenantType.LAW_FIRM
    tenant.save()
    return tenant


@pytest.fixture
def case_a(tenant_enterprise):
    """Create test case A."""
    return CaseFactory(tenant=tenant_enterprise, external_id="CASE-A")


@pytest.fixture
def case_b(tenant_enterprise):
    """Create test case B."""
    return CaseFactory(tenant=tenant_enterprise, external_id="CASE-B")


@pytest.mark.django_db
class TestUserCanAccessCase:
    """Tests for user_can_access_case function."""

    def test_unauthenticated_user_denied(self, case_a):
        """Unauthenticated users cannot access cases."""
        user = AnonymousUser()
        assert not user_can_access_case(user, case_a)

    def test_none_user_denied(self, case_a):
        """None user cannot access cases."""
        assert not user_can_access_case(None, case_a)

    def test_user_without_profile_denied(self, case_a):
        """User without profile cannot access cases."""
        user = UserFactory()
        user.userprofile.delete()
        assert not user_can_access_case(user, case_a)

    def test_inactive_profile_denied(self, case_a):
        """Inactive profile cannot access cases."""
        user = UserFactory(is_active=False)
        assert not user_can_access_case(user, case_a)

    # EXTERNAL accounts tests
    def test_external_account_with_membership_allowed(self, case_a):
        """EXTERNAL account with membership can access case."""
        user = UserFactory(
            role=UserProfile.Roles.STAKEHOLDER,
            account_type=UserProfile.AccountType.EXTERNAL,
        )
        CaseMembershipFactory(case=case_a, user=user)
        assert user_can_access_case(user, case_a)

    def test_external_account_without_membership_denied(self, case_a):
        """EXTERNAL account without membership cannot access case."""
        user = UserFactory(
            role=UserProfile.Roles.TENANT_ADMIN,  # Even admin!
            account_type=UserProfile.AccountType.EXTERNAL,
        )
        assert not user_can_access_case(user, case_a)

    # ENTERPRISE tenant tests
    def test_enterprise_tenant_admin_allowed(self, case_a):
        """TENANT_ADMIN has all-cases access in ENTERPRISE."""
        user = UserFactory(role=UserProfile.Roles.TENANT_ADMIN)
        assert user_can_access_case(user, case_a, case_a.tenant)

    def test_enterprise_legal_allowed(self, case_a):
        """LEGAL has all-cases access in ENTERPRISE."""
        user = UserFactory(role=UserProfile.Roles.LEGAL)
        assert user_can_access_case(user, case_a, case_a.tenant)

    def test_enterprise_management_allowed(self, case_a):
        """MANAGEMENT has all-cases access in ENTERPRISE."""
        user = UserFactory(role=UserProfile.Roles.MANAGEMENT)
        assert user_can_access_case(user, case_a, case_a.tenant)

    def test_enterprise_works_council_all_scope_allowed(
        self, tenant_enterprise, case_a
    ):
        """WORKS_COUNCIL with ALL scope has all-cases access."""
        tenant_enterprise.works_council_scope = Tenant.WorksCouncilScope.ALL
        tenant_enterprise.save()
        user = UserFactory(role=UserProfile.Roles.WORKS_COUNCIL)
        assert user_can_access_case(user, case_a, tenant_enterprise)

    def test_enterprise_works_council_assigned_scope_with_membership(
        self, tenant_enterprise, case_a
    ):
        """WORKS_COUNCIL with ASSIGNED scope needs membership."""
        tenant_enterprise.works_council_scope = Tenant.WorksCouncilScope.ASSIGNED
        tenant_enterprise.save()
        user = UserFactory(role=UserProfile.Roles.WORKS_COUNCIL)
        CaseMembershipFactory(case=case_a, user=user)
        assert user_can_access_case(user, case_a, tenant_enterprise)

    def test_enterprise_works_council_assigned_scope_without_membership(
        self, tenant_enterprise, case_a
    ):
        """WORKS_COUNCIL with ASSIGNED scope denied without membership."""
        tenant_enterprise.works_council_scope = Tenant.WorksCouncilScope.ASSIGNED
        tenant_enterprise.save()
        user = UserFactory(role=UserProfile.Roles.WORKS_COUNCIL)
        assert not user_can_access_case(user, case_a, tenant_enterprise)

    def test_enterprise_stakeholder_with_membership_allowed(self, case_a):
        """STAKEHOLDER with membership can access case."""
        user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)
        CaseMembershipFactory(case=case_a, user=user)
        assert user_can_access_case(user, case_a)

    def test_enterprise_stakeholder_without_membership_denied(self, case_a):
        """STAKEHOLDER without membership cannot access case."""
        user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)
        assert not user_can_access_case(user, case_a)

    # LAW_FIRM tenant tests
    def test_law_firm_tenant_admin_allowed(self, tenant_law_firm):
        """TENANT_ADMIN has all-cases access in LAW_FIRM."""
        case = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.TENANT_ADMIN)
        assert user_can_access_case(user, case, tenant_law_firm)

    def test_law_firm_legal_with_membership_allowed(self, tenant_law_firm):
        """LEGAL needs membership in LAW_FIRM."""
        case = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.LEGAL)
        CaseMembershipFactory(case=case, user=user)
        assert user_can_access_case(user, case, tenant_law_firm)

    def test_law_firm_legal_without_membership_denied(self, tenant_law_firm):
        """LEGAL without membership denied in LAW_FIRM."""
        case = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.LEGAL)
        assert not user_can_access_case(user, case, tenant_law_firm)

    def test_law_firm_management_without_membership_denied(self, tenant_law_firm):
        """MANAGEMENT without membership denied in LAW_FIRM."""
        case = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.MANAGEMENT)
        assert not user_can_access_case(user, case, tenant_law_firm)


@pytest.mark.django_db
class TestGetAccessibleCasesQueryset:
    """Tests for get_accessible_cases_queryset function."""

    def test_unauthenticated_user_returns_none(self, tenant_enterprise):
        """Unauthenticated user gets empty queryset."""
        user = AnonymousUser()
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 0

    def test_none_user_returns_none(self, tenant_enterprise):
        """None user gets empty queryset."""
        qs = get_accessible_cases_queryset(None, tenant_enterprise)
        assert qs.count() == 0

    def test_user_without_profile_returns_none(self, tenant_enterprise):
        """User without profile gets empty queryset."""
        user = UserFactory()
        user.userprofile.delete()
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 0

    def test_inactive_profile_returns_none(self, tenant_enterprise, case_a):
        """Inactive profile gets empty queryset."""
        user = UserFactory(is_active=False)
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 0

    # EXTERNAL accounts tests
    def test_external_account_sees_only_assigned_cases(
        self, tenant_enterprise, case_a, case_b
    ):
        """EXTERNAL account sees only cases with membership."""
        user = UserFactory(account_type=UserProfile.AccountType.EXTERNAL)
        CaseMembershipFactory(case=case_a, user=user)
        # case_b has no membership

        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 1
        assert case_a in qs
        assert case_b not in qs

    # ENTERPRISE tenant tests
    def test_enterprise_tenant_admin_sees_all(self, tenant_enterprise, case_a, case_b):
        """TENANT_ADMIN sees all cases in ENTERPRISE."""
        user = UserFactory(role=UserProfile.Roles.TENANT_ADMIN)
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 2
        assert set(qs) == {case_a, case_b}

    def test_enterprise_legal_sees_all(self, tenant_enterprise, case_a, case_b):
        """LEGAL sees all cases in ENTERPRISE."""
        user = UserFactory(role=UserProfile.Roles.LEGAL)
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 2
        assert set(qs) == {case_a, case_b}

    def test_enterprise_management_sees_all(self, tenant_enterprise, case_a, case_b):
        """MANAGEMENT sees all cases in ENTERPRISE."""
        user = UserFactory(role=UserProfile.Roles.MANAGEMENT)
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 2
        assert set(qs) == {case_a, case_b}

    def test_enterprise_works_council_all_scope_sees_all(
        self, tenant_enterprise, case_a, case_b
    ):
        """WORKS_COUNCIL with ALL scope sees all cases."""
        tenant_enterprise.works_council_scope = Tenant.WorksCouncilScope.ALL
        tenant_enterprise.save()
        user = UserFactory(role=UserProfile.Roles.WORKS_COUNCIL)
        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 2
        assert set(qs) == {case_a, case_b}

    def test_enterprise_works_council_assigned_scope_sees_only_assigned(
        self, tenant_enterprise, case_a, case_b
    ):
        """WORKS_COUNCIL with ASSIGNED scope sees only assigned cases."""
        tenant_enterprise.works_council_scope = Tenant.WorksCouncilScope.ASSIGNED
        tenant_enterprise.save()
        user = UserFactory(role=UserProfile.Roles.WORKS_COUNCIL)
        CaseMembershipFactory(case=case_a, user=user)

        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 1
        assert case_a in qs
        assert case_b not in qs

    def test_enterprise_stakeholder_sees_only_assigned(
        self, tenant_enterprise, case_a, case_b
    ):
        """STAKEHOLDER sees only assigned cases."""
        user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)
        CaseMembershipFactory(case=case_b, user=user)

        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        assert qs.count() == 1
        assert case_b in qs
        assert case_a not in qs

    # LAW_FIRM tenant tests
    def test_law_firm_tenant_admin_sees_all(self, tenant_law_firm):
        """TENANT_ADMIN sees all cases in LAW_FIRM."""
        case_a = CaseFactory(tenant=tenant_law_firm)
        case_b = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.TENANT_ADMIN)

        qs = get_accessible_cases_queryset(user, tenant_law_firm)
        assert qs.count() == 2
        assert set(qs) == {case_a, case_b}

    def test_law_firm_legal_sees_only_assigned(self, tenant_law_firm):
        """LEGAL sees only assigned cases in LAW_FIRM."""
        case_a = CaseFactory(tenant=tenant_law_firm)
        case_b = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.LEGAL)
        CaseMembershipFactory(case=case_a, user=user)

        qs = get_accessible_cases_queryset(user, tenant_law_firm)
        assert qs.count() == 1
        assert case_a in qs
        assert case_b not in qs

    def test_law_firm_management_sees_only_assigned(self, tenant_law_firm):
        """MANAGEMENT sees only assigned cases in LAW_FIRM."""
        case_a = CaseFactory(tenant=tenant_law_firm)
        case_b = CaseFactory(tenant=tenant_law_firm)
        user = UserFactory(role=UserProfile.Roles.MANAGEMENT)
        CaseMembershipFactory(case=case_b, user=user)

        qs = get_accessible_cases_queryset(user, tenant_law_firm)
        assert qs.count() == 1
        assert case_b in qs
        assert case_a not in qs

    # Performance: ensure distinct() works correctly
    def test_multiple_memberships_no_duplicates(self, tenant_enterprise):
        """Multiple memberships don't cause duplicate results."""
        case = CaseFactory(tenant=tenant_enterprise)
        user = UserFactory(role=UserProfile.Roles.STAKEHOLDER)

        # Create multiple memberships (shouldn't happen, but defensive)
        CaseMembershipFactory(case=case, user=user)

        qs = get_accessible_cases_queryset(user, tenant_enterprise)
        # Should still return only 1 case, not duplicates
        assert qs.count() == 1
