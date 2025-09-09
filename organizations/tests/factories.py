import factory
from factory.django import DjangoModelFactory

from organizations.models import OrgMembership, Organization
from users.tests.factories import UserFactory


class OrganizationFactory(DjangoModelFactory):
    class Meta:
        model = Organization

    name = factory.Sequence(lambda n: f"Organization {n}")
    slug = factory.Sequence(lambda n: f"organization-{n}")


class OrgMembershipFactory(DjangoModelFactory):
    class Meta:
        model = OrgMembership

    organization = factory.SubFactory(OrganizationFactory)
    user = factory.SubFactory(UserFactory)
    role = OrgMembership.Role.MEMBER
