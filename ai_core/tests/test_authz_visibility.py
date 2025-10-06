from __future__ import annotations

import pytest
from django.contrib.auth.models import AnonymousUser
from django.core.exceptions import PermissionDenied
from django.test import RequestFactory
from django.test.utils import override_settings

from ai_core.authz.visibility import (
    allow_extended_visibility,
    enforce_visibility_permission,
    require_extended_visibility,
    resolve_effective_visibility,
)
from ai_core.rag.visibility import Visibility
from organizations.models import OrgMembership
from organizations.tests.factories import OrgMembershipFactory, OrganizationFactory
from organizations.utils import set_current_organization
from profiles.models import UserProfile
from users.tests.factories import UserFactory


@pytest.fixture
def request_factory() -> RequestFactory:
    return RequestFactory()


def _build_request(factory: RequestFactory, user) -> object:
    request = factory.get("/rag/search/")
    request.user = user
    return request


@pytest.mark.django_db
def test_allow_extended_visibility_denies_anonymous(request_factory: RequestFactory) -> None:
    request = _build_request(request_factory, AnonymousUser())
    assert allow_extended_visibility(request) is False


@pytest.mark.django_db
def test_allow_extended_visibility_denies_inactive_profile(
    request_factory: RequestFactory,
) -> None:
    user = UserFactory()
    profile = user.userprofile
    profile.role = UserProfile.Roles.ADMIN
    profile.is_active = False
    profile.save(update_fields=["role", "is_active"])
    request = _build_request(request_factory, user)
    assert allow_extended_visibility(request) is False


@pytest.mark.django_db
def test_allow_extended_visibility_accepts_admin_membership(
    request_factory: RequestFactory,
) -> None:
    user = UserFactory()
    profile = user.userprofile
    profile.role = UserProfile.Roles.GUEST
    profile.is_active = True
    profile.save(update_fields=["role", "is_active"])
    organization = OrganizationFactory()
    OrgMembershipFactory(
        organization=organization,
        user=user,
        role=OrgMembership.Role.ADMIN,
    )
    request = _build_request(request_factory, user)
    request.organization = organization
    with set_current_organization(organization):
        assert allow_extended_visibility(request) is True


@pytest.mark.django_db
@override_settings(RAG_INTERNAL_KEYS=["service-key"])
def test_allow_extended_visibility_accepts_service_key(
    request_factory: RequestFactory,
) -> None:
    user = UserFactory()
    profile = user.userprofile
    profile.role = UserProfile.Roles.GUEST
    profile.is_active = True
    profile.save(update_fields=["role", "is_active"])
    request = _build_request(request_factory, user)
    request.META["HTTP_X_INTERNAL_KEY"] = "service-key"
    assert allow_extended_visibility(request) is True


@pytest.mark.django_db
def test_resolve_effective_visibility_normalises_to_active_on_denial(
    request_factory: RequestFactory,
) -> None:
    request = _build_request(request_factory, AnonymousUser())
    visibility = resolve_effective_visibility(request, "deleted")
    assert visibility is Visibility.ACTIVE


@pytest.mark.django_db
def test_resolve_effective_visibility_defaults_to_active(
    request_factory: RequestFactory,
) -> None:
    user = UserFactory()
    request = _build_request(request_factory, user)
    visibility = resolve_effective_visibility(request, None)
    assert visibility is Visibility.ACTIVE


@pytest.mark.django_db
def test_resolve_effective_visibility_preserves_when_allowed(
    request_factory: RequestFactory,
) -> None:
    user = UserFactory()
    profile = user.userprofile
    profile.role = UserProfile.Roles.ADMIN
    profile.is_active = True
    profile.save(update_fields=["role", "is_active"])
    assert profile.role == UserProfile.Roles.ADMIN
    request = _build_request(request_factory, user)
    assert allow_extended_visibility(request) is True
    visibility = resolve_effective_visibility(request, "all")
    assert visibility is Visibility.ALL


@pytest.mark.django_db
def test_enforce_visibility_permission_raises_forbidden_when_denied(
    request_factory: RequestFactory,
) -> None:
    request = _build_request(request_factory, AnonymousUser())
    with pytest.raises(PermissionDenied):
        enforce_visibility_permission(request, "deleted")


@pytest.mark.django_db
def test_require_extended_visibility_decorator_rejects_when_forbidden(
    request_factory: RequestFactory,
) -> None:
    @require_extended_visibility
    def _view(request, *, visibility=None):
        return "ok"

    request = _build_request(request_factory, AnonymousUser())
    with pytest.raises(PermissionDenied):
        _view(request, visibility="all")


@pytest.mark.django_db
def test_require_extended_visibility_decorator_allows_when_permitted(
    request_factory: RequestFactory,
) -> None:
    user = UserFactory()
    profile = user.userprofile
    profile.role = UserProfile.Roles.ADMIN
    profile.is_active = True
    profile.save(update_fields=["role", "is_active"])

    @require_extended_visibility
    def _view(request, *, visibility=None):
        return "ok"

    request = _build_request(request_factory, user)
    assert user.userprofile.role == UserProfile.Roles.ADMIN
    assert allow_extended_visibility(request) is True
    assert _view(request, visibility="all") == "ok"
