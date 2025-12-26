"""Factory fixtures for cases app tests."""

import factory
from factory.django import DjangoModelFactory

from cases.models import Case, CaseMembership
from users.tests.factories import UserFactory


class CaseFactory(DjangoModelFactory):
    """Factory for Case model."""

    class Meta:
        model = Case

    external_id = factory.Sequence(lambda n: f"CASE-{n:04d}")
    title = factory.Faker("sentence", nb_words=4)
    status = Case.Status.OPEN
    phase = ""
    metadata = factory.Dict({})


class CaseMembershipFactory(DjangoModelFactory):
    """Factory for CaseMembership model."""

    class Meta:
        model = CaseMembership

    case = factory.SubFactory(CaseFactory)
    user = factory.SubFactory(UserFactory)
    granted_by = factory.SubFactory(UserFactory)
