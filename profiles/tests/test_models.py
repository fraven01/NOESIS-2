import pytest
from django_tenants.utils import schema_context

from profiles.models import UserProfile
from profiles.services import ensure_user_profile
from users.tests.factories import UserFactory


pytestmark = [
    pytest.mark.slow,
    pytest.mark.django_db,
    pytest.mark.xdist_group("tenant_ops"),
]


def test_user_profile_isolation(tenant_pool):
    tenant1 = tenant_pool["alpha"]
    tenant2 = tenant_pool["beta"]

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
