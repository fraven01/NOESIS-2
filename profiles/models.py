from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    class Roles(models.TextChoices):
        TENANT_ADMIN = "TENANT_ADMIN", "Tenant Admin"
        LEGAL = "LEGAL", "Legal"
        WORKS_COUNCIL = "WORKS_COUNCIL", "Works Council"
        MANAGEMENT = "MANAGEMENT", "Management"
        STAKEHOLDER = "STAKEHOLDER", "Stakeholder"

    class AccountType(models.TextChoices):
        INTERNAL = "INTERNAL", "Internal"
        EXTERNAL = "EXTERNAL", "External"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    role = models.CharField(
        max_length=20, choices=Roles.choices, default=Roles.STAKEHOLDER
    )
    account_type = models.CharField(
        max_length=20,
        choices=AccountType.choices,
        default=AccountType.INTERNAL,
    )
    expires_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.user} ({self.role})"
