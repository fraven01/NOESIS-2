from django.db import connection

from .models import UserProfile


def schema_name():
    """Return the current connection's schema name."""
    return connection.schema_name


def ensure_user_profile(user):
    """Ensure a user has a profile in the current tenant schema."""
    schema_name()
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile
