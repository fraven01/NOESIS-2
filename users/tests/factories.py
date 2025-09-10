import secrets

import factory
from django.utils import timezone
from factory.django import DjangoModelFactory

from profiles.models import UserProfile
from users.models import Invitation, User


class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda o: f"{o.username}@example.com")


class InvitationFactory(DjangoModelFactory):
    class Meta:
        model = Invitation

    email = factory.Faker("email")
    role = UserProfile.Roles.GUEST
    token = factory.LazyFunction(lambda: secrets.token_urlsafe(16))
    expires_at = factory.LazyFunction(
        lambda: timezone.now() + timezone.timedelta(days=7)
    )
