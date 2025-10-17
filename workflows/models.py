from django.db import models

from common.models import TimestampedModel
from organizations.models import Organization
from organizations.query import OrganizationManager


class WorkflowTemplate(TimestampedModel):
    """A collection of ordered steps representing a document workflow template."""

    name = models.CharField(max_length=255)

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return self.name


class WorkflowStep(TimestampedModel):
    """Single step within a :class:`WorkflowTemplate`."""

    template = models.ForeignKey(
        WorkflowTemplate, related_name="steps", on_delete=models.CASCADE
    )
    order = models.PositiveIntegerField()
    instructions = models.TextField()

    class Meta:
        ordering = ["order"]

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.template} - step {self.order}"


class WorkflowInstance(TimestampedModel):
    """Tracks the workflow status for an :class:`Organization`."""

    STATUS_DRAFT = "draft"
    STATUS_REVIEW = "review"
    STATUS_FINAL = "final"

    STATUS_CHOICES = [
        (STATUS_DRAFT, "Draft"),
        (STATUS_REVIEW, "Review"),
        (STATUS_FINAL, "Final"),
    ]

    organization = models.ForeignKey(
        Organization, related_name="workflow_instances", on_delete=models.CASCADE
    )
    status = models.CharField(
        max_length=10, choices=STATUS_CHOICES, default=STATUS_DRAFT
    )
    objects = OrganizationManager()

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return f"Workflow {self.get_status_display()} for {self.organization}"
