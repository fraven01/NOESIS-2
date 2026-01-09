import pytest
from django.conf import settings
from django.db import connection
from django_tenants.utils import schema_context

from profiles.models import UserProfile
from profiles.signals import create_user_profile
from users.tests.factories import UserFactory


pytestmark = [
    pytest.mark.slow,
    pytest.mark.django_db,
    pytest.mark.xdist_group("tenant_ops"),
]


def test_signal_skips_public_schema(monkeypatch, tenant_pool):
    tenant = tenant_pool["alpha"]
    with schema_context(tenant.schema_name):
        user = UserFactory()
        UserProfile.objects.all().delete()
        monkeypatch.setattr(connection, "schema_name", settings.PUBLIC_SCHEMA_NAME)
        create_user_profile(sender=None, instance=user, created=True)
        # Ensure we evaluate the assertion in the tenant schema context
        with schema_context(tenant.schema_name):
            assert not UserProfile.objects.filter(user=user).exists()


def test_signal_creates_profile_per_tenant(tenant_pool):
    tenant1 = tenant_pool["alpha"]
    tenant2 = tenant_pool["beta"]
    with schema_context(tenant1.schema_name):
        user1 = UserFactory()
    with schema_context(tenant2.schema_name):
        user2 = UserFactory()
    with schema_context(tenant1.schema_name):
        assert UserProfile.objects.filter(user=user1).exists()
        assert not UserProfile.objects.filter(user=user2).exists()
    with schema_context(tenant2.schema_name):
        assert UserProfile.objects.filter(user=user2).exists()
