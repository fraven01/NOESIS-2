from django_tenants.test.cases import TenantTestCase
from django_tenants.utils import schema_context

from customers.tests.factories import TenantFactory
from profiles.models import UserProfile
from profiles.services import ensure_user_profile
from users.tests.factories import UserFactory


class TestUserProfileIsolation(TenantTestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = UserFactory()
        cls.tenant1 = TenantFactory(schema_name="alpha")
        cls.tenant2 = TenantFactory(schema_name="beta")

    def test_profile_is_schema_isolated(self):
        with schema_context(self.tenant1.schema_name):
            ensure_user_profile(self.user)
            self.assertEqual(UserProfile.objects.filter(user=self.user).count(), 1)
        with schema_context(self.tenant2.schema_name):
            self.assertFalse(UserProfile.objects.filter(user=self.user).exists())
            ensure_user_profile(self.user)
            self.assertEqual(UserProfile.objects.filter(user=self.user).count(), 1)
        with schema_context(self.tenant1.schema_name):
            self.assertEqual(UserProfile.objects.filter(user=self.user).count(), 1)
