from django.conf import settings
from django.db import models

from common.models import TimestampedModel
from projects.models import Project
from organizations.query import OrganizationManager


class DocumentType(TimestampedModel):
    """Type of document supported by the system."""

    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return self.name


class Document(TimestampedModel):
    """An uploaded document belonging to a user."""

    STATUS_UPLOADED = "uploaded"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_CHOICES = [
        (STATUS_UPLOADED, "hochgeladen"),
        (STATUS_PROCESSING, "in Bearbeitung"),
        (STATUS_COMPLETED, "abgeschlossen"),
    ]

    file = models.FileField(upload_to="documents/")
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_UPLOADED
    )
    type = models.ForeignKey(
        DocumentType, related_name="documents", on_delete=models.CASCADE
    )
    project = models.ForeignKey(
        Project, related_name="documents", on_delete=models.CASCADE
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="documents", on_delete=models.CASCADE
    )
    objects = OrganizationManager(organization_field="project__organization")

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.type.name} ({self.get_status_display()})"
