import pytest

from .factories import UserFactory
from documents.tests.factories import DocumentFactory


@pytest.mark.django_db
def test_user_creation():
    user = UserFactory(email="test@example.com")
    assert user.email == "test@example.com"


@pytest.mark.django_db
def test_user_document_relationship():
    user = UserFactory()
    document = DocumentFactory(owner=user)
    assert document.owner == user
