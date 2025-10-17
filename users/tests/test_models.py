import pytest

from .factories import UserFactory


@pytest.mark.django_db
def test_user_creation():
    user = UserFactory(email="test@example.com")
    assert user.email == "test@example.com"
