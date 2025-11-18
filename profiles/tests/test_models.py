import pytest
from django_tenants.utils import schema_context

from customers.tests.factories import TenantFactory
from profiles.models import UserProfile
from profiles.services import ensure_user_profile
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def test_user_profile_isolation():
    tenant1 = TenantFactory(schema_name="alpha")
    tenant2 = TenantFactory(schema_name="beta")

    with schema_context(tenant1.schema_name):
        user1 = UserFactory()
        assert UserProfile.objects.filter(user=user1).count() == 1
        ensure_user_profile(user1)
        assert UserProfile.objects.filter(user=user1).count() == 1

    with schema_context(tenant2.schema_name):
        user2 = UserFactory()
        assert UserProfile.objects.filter(user=user2).count() == 1
        ensure_user_profile(user2)
        assert UserProfile.objects.filter(user=user2).count() == 1

    with schema_context(tenant1.schema_name):
        assert UserProfile.objects.count() == 1
