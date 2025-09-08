from django.conf import settings
from django.db import models

from common.models import TimestampedModel
from workflows.models import Workflow


class DocumentType(TimestampedModel):
    """Type of document supported by the system."""

    name = models.CharField(max_length=255)
    description = models.TextField()
    workflow = models.ForeignKey(
        Workflow, related_name="document_types", on_delete=models.CASCADE
    )

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
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="documents", on_delete=models.CASCADE
    )

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.type.name} ({self.get_status_display()})"
