import pytest
from django.core.exceptions import ValidationError

from customers.models import Domain
from .factories import DomainFactory


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_only_one_primary_domain_per_tenant(tenant_pool):
    tenant = tenant_pool["alpha"]
    d1 = DomainFactory(tenant=tenant, domain="one.example.com", is_primary=True)
    d2 = DomainFactory(tenant=tenant, domain="two.example.com", is_primary=True)

    d1.refresh_from_db()
    d2.refresh_from_db()

    assert Domain.objects.filter(tenant=tenant, is_primary=True).count() == 1
    assert d1.is_primary != d2.is_primary


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_tenant_accepts_case_lifecycle_definition(tenant_pool):
    tenant = tenant_pool["beta"]
    tenant.case_lifecycle_definition = {
        "phases": ["anzeige"],
        "transitions": [
            {
                "from_phase": None,
                "to_phase": "anzeige",
                "trigger_events": ["ingestion_run_queued"],
            }
        ],
    }

    tenant.full_clean()


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_tenant_rejects_invalid_case_lifecycle_definition(tenant_pool):
    tenant = tenant_pool["gamma"]
    tenant.case_lifecycle_definition = {"phases": [""], "transitions": [{}]}

    with pytest.raises(ValidationError):
        tenant.full_clean()
