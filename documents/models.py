"""Database models supporting document lifecycle persistence."""

from __future__ import annotations

from django.db import models


class DocumentLifecycleState(models.Model):
    """Latest lifecycle status for a document within a tenant workflow."""

    tenant_id = models.CharField(max_length=255)
    document_id = models.UUIDField()
    workflow_id = models.CharField(max_length=255, blank=True, default="")
    state = models.CharField(max_length=32)
    changed_at = models.DateTimeField()
    reason = models.TextField(blank=True, default="")
    policy_events = models.JSONField(blank=True, default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant_id", "document_id", "workflow_id"),
                name="document_lifecycle_unique_record",
            )
        ]
        indexes = [
            models.Index(fields=("tenant_id", "workflow_id")),
        ]


class DocumentIngestionRun(models.Model):
    """Persistent metadata for the last ingestion run per tenant/case."""

    tenant = models.CharField(max_length=255)
    case = models.CharField(max_length=255)
    run_id = models.CharField(max_length=255)
    status = models.CharField(max_length=32)
    queued_at = models.CharField(max_length=64)
    started_at = models.CharField(max_length=64, blank=True, default="")
    finished_at = models.CharField(max_length=64, blank=True, default="")
    duration_ms = models.FloatField(null=True, blank=True)
    inserted_documents = models.IntegerField(null=True, blank=True)
    replaced_documents = models.IntegerField(null=True, blank=True)
    skipped_documents = models.IntegerField(null=True, blank=True)
    inserted_chunks = models.IntegerField(null=True, blank=True)
    invalid_document_ids = models.JSONField(blank=True, default=list)
    document_ids = models.JSONField(blank=True, default=list)
    trace_id = models.CharField(max_length=255, blank=True, default="")
    embedding_profile = models.CharField(max_length=255, blank=True, default="")
    source = models.CharField(max_length=255, blank=True, default="")
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "case"), name="document_ingestion_run_unique_case"
            )
        ]
        indexes = [
            models.Index(fields=("tenant", "case")),
            models.Index(fields=("tenant", "run_id")),
        ]
