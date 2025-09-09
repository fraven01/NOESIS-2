import uuid

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _

from common.models import TimestampedModel


class Organization(TimestampedModel):
    """Represents an organization of users."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return self.name


class OrgMembership(TimestampedModel):
    """Links a :class:`User` to an :class:`Organization` with a role."""

    class Role(models.TextChoices):
        MEMBER = "member", _("Member")
        ADMIN = "admin", _("Admin")

    organization = models.ForeignKey(
        Organization, related_name="memberships", on_delete=models.CASCADE
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name="org_memberships",
        on_delete=models.CASCADE,
    )
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.MEMBER)

    class Meta:
        unique_together = ("organization", "user")

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.user} in {self.organization} ({self.get_role_display()})"
