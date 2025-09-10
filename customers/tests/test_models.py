import pytest

from customers.models import Domain
from .factories import DomainFactory, TenantFactory


@pytest.mark.django_db
def test_only_one_primary_domain_per_tenant():
    tenant = TenantFactory()
    d1 = DomainFactory(tenant=tenant, domain="one.example.com", is_primary=True)
    d2 = DomainFactory(tenant=tenant, domain="two.example.com", is_primary=True)

    d1.refresh_from_db()
    d2.refresh_from_db()

    assert Domain.objects.filter(tenant=tenant, is_primary=True).count() == 1
    assert d1.is_primary != d2.is_primary
