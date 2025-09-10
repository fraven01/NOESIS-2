import factory
from factory.django import DjangoModelFactory

from profiles.models import UserProfile
from users.tests.factories import UserFactory


class UserProfileFactory(DjangoModelFactory):
    class Meta:
        model = UserProfile

    user = factory.SubFactory(UserFactory)
    role = UserProfile.Roles.GUEST
