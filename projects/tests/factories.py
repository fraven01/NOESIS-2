import factory
from factory.django import DjangoModelFactory

from projects.models import Project
from users.tests.factories import UserFactory
from organizations.tests.factories import OrganizationFactory


class ProjectFactory(DjangoModelFactory):
    class Meta:
        model = Project

    name = factory.Sequence(lambda n: f"Project {n}")
    description = factory.Faker("sentence")
    owner = factory.SubFactory(UserFactory)
    organization = factory.SubFactory(OrganizationFactory)
