import pytest

from customers.models import Tenant
from users.tests.factories import UserFactory


@pytest.fixture
def user():
    """Create a staff user with a profile for theme tests."""
    return UserFactory(is_staff=True)


@pytest.fixture
def tenant(test_tenant_schema_name):
    """Provide the tenant that pytest-django bootstrapped for theme tests."""
    return Tenant.objects.get(schema_name=test_tenant_schema_name)


@pytest.fixture
def auth_client(client, user):
    """Logged-in client that mimics authenticated interactions in theme views."""
    client.force_login(user)
    return client
