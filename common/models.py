from django.db import models


class TimestampedModel(models.Model):
    """Abstract base model with self-updating ``created_at`` and ``updated_at``.

    ``created_at`` is set when the object is created and ``updated_at`` is
    refreshed on each save.
    """

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

