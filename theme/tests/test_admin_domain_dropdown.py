import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from customers.tests.factories import DomainFactory


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_domain_dropdown_shows_domains(client, tenant_pool):
    User = get_user_model()
    user = User.objects.create_user(
        username="admin", password="pass", is_staff=True, is_superuser=True
    )
    client.force_login(user)
    DomainFactory(tenant=tenant_pool["alpha"], domain="alpha.example.com")
    DomainFactory(tenant=tenant_pool["beta"], domain="beta.example.com")
    response = client.get(reverse("admin:index"))
    assert response.status_code == 200
    content = response.content.decode()
    assert 'id="domain-switcher"' in content
    assert "alpha.example.com" in content
    assert "beta.example.com" in content
