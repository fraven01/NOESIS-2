import pytest
from django.http import HttpResponse
from django.test import override_settings
from django.urls import path

from organizations.guards import require_org_member
from organizations.tests.factories import OrgMembershipFactory, OrganizationFactory
from organizations.utils import set_current_organization
from users.tests.factories import UserFactory


@require_org_member
def sample_view(request):
    return HttpResponse("ok")


urlpatterns = [path("sample/", sample_view, name="sample")]


@override_settings(ROOT_URLCONF=__name__)
@pytest.mark.django_db
def test_member_can_modify(client):
    org = OrganizationFactory()
    user = UserFactory()
    OrgMembershipFactory(organization=org, user=user)
    client.force_login(user)
    with set_current_organization(org):
        response = client.post("/sample/")
    assert response.status_code == 200


@override_settings(ROOT_URLCONF=__name__)
@pytest.mark.django_db
def test_non_member_receives_403(client):
    org = OrganizationFactory()
    user = UserFactory()
    client.force_login(user)
    with set_current_organization(org):
        response = client.post("/sample/")
    assert response.status_code == 403


@override_settings(ROOT_URLCONF=__name__)
@pytest.mark.django_db
def test_non_member_cannot_read(client):
    org = OrganizationFactory()
    user = UserFactory()
    client.force_login(user)
    with set_current_organization(org):
        response = client.get("/sample/")
    assert response.status_code == 403
