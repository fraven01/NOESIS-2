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

    class DocumentViewMode(models.TextChoices):
        LIST = "LIST", "List"
        GRID = "GRID", "Grid"
        TABLE = "TABLE", "Table"

    class ExternalEmailFrequency(models.TextChoices):
        IMMEDIATE = "IMMEDIATE", "Immediate"
        DAILY = "DAILY", "Daily"

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
    document_view_mode = models.CharField(
        max_length=10,
        choices=DocumentViewMode.choices,
        default=DocumentViewMode.LIST,
    )
    documents_per_page = models.IntegerField(default=25)
    notify_on_document_upload = models.BooleanField(default=True)
    notify_on_mention = models.BooleanField(default=True)
    notify_on_comment_reply = models.BooleanField(default=True)
    notify_on_case_document = models.BooleanField(default=False)
    external_email_enabled = models.BooleanField(default=False)
    external_email_frequency = models.CharField(
        max_length=20,
        choices=ExternalEmailFrequency.choices,
        default=ExternalEmailFrequency.IMMEDIATE,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.user} ({self.role})"
