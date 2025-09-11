import datetime

import factory
from factory.django import DjangoModelFactory

from customers.models import Domain, Tenant


class TenantFactory(DjangoModelFactory):
    class Meta:
        model = Tenant
        django_get_or_create = ("schema_name",)

    schema_name = factory.Sequence(lambda n: f"tenant{n}")
    name = factory.Sequence(lambda n: f"Tenant {n}")
    paid_until = factory.LazyFunction(lambda: datetime.date.today())
    on_trial = True

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        # Tenants must be created from the public schema in django-tenants.
        from django_tenants.utils import schema_context, get_public_schema_name

        with schema_context(get_public_schema_name()):
            obj = super()._create(model_class, *args, **kwargs)
        return obj


class DomainFactory(DjangoModelFactory):
    class Meta:
        model = Domain

    tenant = factory.SubFactory(TenantFactory)
    domain = factory.Sequence(lambda n: f"domain{n}.example.com")
    is_primary = True
