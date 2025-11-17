"""Case lifecycle models."""

from __future__ import annotations

import uuid

from django.db import models


class Case(models.Model):
    """Primary case record for a tenant workflow."""

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        CLOSED = "closed", "Closed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="cases",
    )
    external_id = models.CharField(max_length=255)
    title = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.OPEN,
    )
    phase = models.CharField(max_length=64, blank=True, default="")
    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    closed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "external_id"),
                name="case_unique_external_id",
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant", "status"),
                name="case_tenant_status_idx",
            ),
            models.Index(
                fields=("tenant", "external_id"),
                name="case_tenant_external_idx",
            ),
        ]

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.external_id} ({self.status})"


class CaseEvent(models.Model):
    """Lifecycle event tied to a case."""

    case = models.ForeignKey(
        Case,
        on_delete=models.CASCADE,
        related_name="events",
    )
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="case_events",
    )
    event_type = models.CharField(max_length=64)
    source = models.CharField(max_length=128, blank=True, default="")
    graph_name = models.CharField(max_length=128, blank=True, default="")
    ingestion_run = models.ForeignKey(
        "documents.DocumentIngestionRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="case_events",
    )
    workflow_id = models.CharField(max_length=255, blank=True, default="")
    collection_id = models.CharField(max_length=255, blank=True, default="")
    trace_id = models.CharField(max_length=255, blank=True, default="")
    payload = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(
                fields=("tenant", "event_type"),
                name="case_event_tenant_type_idx",
            ),
            models.Index(
                fields=("tenant", "created_at"),
                name="case_event_tenant_created_idx",
            ),
        ]
