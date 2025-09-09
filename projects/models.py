from django.conf import settings
from django.db import models

from common.models import TimestampedModel
from workflows.models import WorkflowTemplate


class Project(TimestampedModel):
    """Represents a software project or initiative."""

    STATUS_INITIATED = "initiated"
    STATUS_NEGOTIATION = "negotiation"
    STATUS_COMPLETED = "completed"
    STATUS_PAUSED = "paused"

    STATUS_CHOICES = [
        (STATUS_INITIATED, "Initiert"),
        (STATUS_NEGOTIATION, "In Verhandlung"),
        (STATUS_COMPLETED, "Abgeschlossen"),
        (STATUS_PAUSED, "Pausiert"),
    ]

    name = models.CharField(max_length=255)
    description = models.TextField()
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_INITIATED
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="projects", on_delete=models.CASCADE
    )
    organization = models.ForeignKey(
        "organizations.Organization", on_delete=models.CASCADE
    )

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return self.name


class WorkflowInstance(TimestampedModel):
    """Concrete workflow state for a specific :class:`Project`."""

    project = models.OneToOneField(
        Project, related_name="workflow", on_delete=models.CASCADE
    )
    template = models.ForeignKey(
        WorkflowTemplate, related_name="instances", on_delete=models.CASCADE
    )
    state = models.JSONField(default=dict, blank=True)

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"Workflow for {self.project}"
