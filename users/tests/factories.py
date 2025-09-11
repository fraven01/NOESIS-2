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

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        # Ensure user IDs are globally unique across tenant schemas in tests to
        # avoid PK collisions when comparing FK relations across schemas.
        from django.db import connection

        schema = getattr(connection, "schema_name", "public") or "public"
        # Derive a large, schema-specific offset to keep sequences disjoint.
        base = (abs(hash(schema)) % 1000 + 1) * 1_000_000
        seq = getattr(cls, "_id_seq", 0) + 1
        cls._id_seq = seq
        kwargs.setdefault("id", base + seq)
        return super()._create(model_class, *args, **kwargs)


class InvitationFactory(DjangoModelFactory):
    class Meta:
        model = Invitation

    email = factory.Faker("email")
    role = UserProfile.Roles.GUEST
    token = factory.LazyFunction(lambda: secrets.token_urlsafe(16))
    expires_at = factory.LazyFunction(
        lambda: timezone.now() + timezone.timedelta(days=7)
    )
