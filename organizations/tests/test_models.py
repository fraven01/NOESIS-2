import pytest
from django.db import IntegrityError

from .factories import OrgMembershipFactory, OrganizationFactory


@pytest.mark.django_db
def test_organization_factory_creates_instance():
    organization = OrganizationFactory()
    assert organization.pk is not None


@pytest.mark.django_db
def test_orgmembership_factory_creates_instance():
    membership = OrgMembershipFactory()
    assert membership.pk is not None


@pytest.mark.django_db
def test_orgmembership_unique_constraint():
    membership = OrgMembershipFactory()
    with pytest.raises(IntegrityError):
        OrgMembershipFactory(organization=membership.organization, user=membership.user)
