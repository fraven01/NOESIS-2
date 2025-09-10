import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from customers.tests.factories import DomainFactory


@pytest.mark.django_db
def test_domain_dropdown_shows_domains(client):
    User = get_user_model()
    user = User.objects.create_user(
        username="admin", password="pass", is_staff=True, is_superuser=True
    )
    client.force_login(user)
    DomainFactory(domain="alpha.example.com")
    DomainFactory(domain="beta.example.com")
    response = client.get(reverse("admin:index"))
    assert response.status_code == 200
    content = response.content.decode()
    assert 'id="domain-switcher"' in content
    assert "alpha.example.com" in content
    assert "beta.example.com" in content
