from django_tenants.test.cases import TenantTestCase
from django_tenants.utils import schema_context

from customers.tests.factories import TenantFactory
from profiles.models import UserProfile
from profiles.services import ensure_user_profile
from users.tests.factories import UserFactory


class TestUserProfileIsolation(TenantTestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tenant1 = TenantFactory(schema_name="alpha")
        cls.tenant1.create_schema(check_if_exists=True)
        cls.tenant2 = TenantFactory(schema_name="beta")
        cls.tenant2.create_schema(check_if_exists=True)

    def test_profile_is_schema_isolated(self):
        # Create a user per-tenant (users are tenant-scoped)
        with schema_context(self.tenant1.schema_name):
            user1 = UserFactory()
            # Profile is created by signal on user creation
            self.assertEqual(UserProfile.objects.filter(user=user1).count(), 1)
            # Idempotent ensure
            ensure_user_profile(user1)
            self.assertEqual(UserProfile.objects.filter(user=user1).count(), 1)
        with schema_context(self.tenant2.schema_name):
            user2 = UserFactory()
            # Signal created the profile for user2 in tenant2
            self.assertEqual(UserProfile.objects.filter(user=user2).count(), 1)
            ensure_user_profile(user2)
            self.assertEqual(UserProfile.objects.filter(user=user2).count(), 1)
        # Verify isolation: tenant1 still has exactly one profile
        with schema_context(self.tenant1.schema_name):
            self.assertEqual(UserProfile.objects.count(), 1)
