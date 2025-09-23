from django.db import models
from django.db.models import Q
from django_tenants.models import DomainMixin, TenantMixin


class Tenant(TenantMixin):
    name = models.CharField(max_length=100)
    paid_until = models.DateField(null=True, blank=True)
    on_trial = models.BooleanField(default=True)
    created_on = models.DateField(auto_now_add=True)
    pii_mode = models.CharField(max_length=32, null=True, blank=True)
    pii_policy = models.CharField(max_length=32, null=True, blank=True)
    pii_logging_redaction = models.BooleanField(null=True, blank=True)
    pii_post_response = models.BooleanField(null=True, blank=True)
    pii_deterministic = models.BooleanField(null=True, blank=True)
    pii_hmac_secret = models.CharField(max_length=512, null=True, blank=True)
    pii_name_detection = models.BooleanField(null=True, blank=True)

    auto_create_schema = True

    def get_pii_config(self) -> dict[str, object]:
        from ai_core.infra.pii_flags import resolve_tenant_pii_config

        return resolve_tenant_pii_config(self)


class Domain(DomainMixin):
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant"],
                condition=Q(is_primary=True),
                name="unique_primary_domain_per_tenant",
            )
        ]
