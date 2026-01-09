import secrets
import uuid

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
        # Extract UserProfile-specific kwargs
        # These are NOT User model fields, so we pop them before user creation
        profile_role = kwargs.pop("role", None)
        profile_account_type = kwargs.pop("account_type", None)
        profile_expires_at = kwargs.pop("expires_at", None)
        # Support both 'is_active' (sets both user and profile) and 'profile_is_active' (profile only)
        # When 'is_active' is used, it sets BOTH user.is_active and profile.is_active
        is_active_from_kwargs = kwargs.get("is_active", None)
        profile_is_active = kwargs.pop("profile_is_active", is_active_from_kwargs)

        kwargs.setdefault("id", uuid.uuid4())

        user = super()._create(model_class, *args, **kwargs)

        # ALWAYS ensure UserProfile exists (required for auth system)
        from profiles.services import ensure_user_profile

        profile = ensure_user_profile(user)

        # Update profile fields if specified, otherwise use model defaults
        # Note: ensure_user_profile already created the profile with model defaults

        needs_save = False

        if profile_role is not None:
            profile.role = profile_role
            needs_save = True

        if profile_account_type is not None:
            profile.account_type = profile_account_type
            needs_save = True

        if profile_is_active is not None:
            profile.is_active = profile_is_active
            needs_save = True

        if profile_expires_at is not None:
            profile.expires_at = profile_expires_at
            needs_save = True

        # Save only if we made changes
        if needs_save:
            profile.save()
            # Refresh user to clear Django's cached reverse OneToOne relation
            # This ensures user.userprofile returns the updated profile
            user.refresh_from_db()
            profile.refresh_from_db()

        return user


class InvitationFactory(DjangoModelFactory):
    class Meta:
        model = Invitation

    email = factory.Faker("email")
    role = UserProfile.Roles.STAKEHOLDER
    account_type = UserProfile.AccountType.INTERNAL
    token = factory.LazyFunction(lambda: secrets.token_urlsafe(16))
    invitation_expires_at = factory.LazyFunction(
        lambda: timezone.now() + timezone.timedelta(days=7)
    )
