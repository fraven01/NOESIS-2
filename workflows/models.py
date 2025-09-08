from django.db import models

from common.models import TimestampedModel


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
