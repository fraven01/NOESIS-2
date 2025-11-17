from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django_tenants.models import DomainMixin, TenantMixin

from pydantic import ValidationError as PydanticValidationError


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
    case_lifecycle_definition = models.JSONField(null=True, blank=True)

    auto_create_schema = True

    def get_pii_config(self) -> dict[str, object]:
        from ai_core.infra.pii_flags import resolve_tenant_pii_config

        return resolve_tenant_pii_config(self)

    def clean(self) -> None:
        super().clean()
        definition = getattr(self, "case_lifecycle_definition", None)
        if definition in (None, "", {}):
            return
        from cases.contracts import parse_case_lifecycle_definition

        try:
            parse_case_lifecycle_definition(definition)
        except PydanticValidationError as exc:  # pragma: no cover - safety
            messages = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error.get("loc", ()))
                message = error.get("msg", "Invalid lifecycle definition")
                if location:
                    messages.append(f"{location}: {message}")
                else:
                    messages.append(str(message))
            if not messages:
                messages = ["Invalid lifecycle definition"]
            raise ValidationError({"case_lifecycle_definition": messages}) from exc


class Domain(DomainMixin):
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant"],
                condition=Q(is_primary=True),
                name="unique_primary_domain_per_tenant",
            )
        ]
