import datetime

import factory
from factory.django import DjangoModelFactory

from customers.models import Tenant


class TenantFactory(DjangoModelFactory):
    class Meta:
        model = Tenant

    schema_name = factory.Sequence(lambda n: f"tenant{n}")
    name = factory.Sequence(lambda n: f"Tenant {n}")
    paid_until = factory.LazyFunction(lambda: datetime.date.today())
    on_trial = True
