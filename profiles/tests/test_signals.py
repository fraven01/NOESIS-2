import pytest
from django.conf import settings
from django.db import connection
from django_tenants.utils import schema_context

from customers.tests.factories import TenantFactory
from profiles.models import UserProfile
from profiles.signals import create_user_profile
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def test_signal_skips_public_schema(monkeypatch):
    tenant = TenantFactory(schema_name="alpha")
    with schema_context(tenant.schema_name):
        user = UserFactory()
        UserProfile.objects.all().delete()
        monkeypatch.setattr(connection, "schema_name", settings.PUBLIC_SCHEMA_NAME)
        create_user_profile(sender=None, instance=user, created=True)
        # Ensure we evaluate the assertion in the tenant schema context
        with schema_context(tenant.schema_name):
            assert not UserProfile.objects.filter(user=user).exists()


def test_signal_creates_profile_per_tenant():
    tenant1 = TenantFactory(schema_name="alpha")
    tenant2 = TenantFactory(schema_name="beta")
    with schema_context(tenant1.schema_name):
        user1 = UserFactory()
    with schema_context(tenant2.schema_name):
        user2 = UserFactory()
    with schema_context(tenant1.schema_name):
        assert UserProfile.objects.filter(user=user1).exists()
        assert not UserProfile.objects.filter(user=user2).exists()
    with schema_context(tenant2.schema_name):
        assert UserProfile.objects.filter(user=user2).exists()
