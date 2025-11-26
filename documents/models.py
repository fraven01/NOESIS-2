"""Database models supporting document lifecycle persistence."""

from __future__ import annotations

import uuid

from django.db import models


from . import framework_models  # noqa: F401


class DocumentCollection(models.Model):
    """Logical document grouping mapping to vector collections."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="document_collections",
    )
    case = models.ForeignKey(
        "cases.Case",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="document_collections",
    )
    name = models.CharField(max_length=255)
    key = models.CharField(max_length=128)
    collection_id = models.UUIDField()
    type = models.CharField(max_length=64, blank=True, default="")
    visibility = models.CharField(max_length=64, blank=True, default="")
    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "key"),
                name="document_collection_unique_key",
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant", "case"),
                name="doc_collection_tenant_case_idx",
            ),
        ]

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.name} ({self.key})"


class DocumentLifecycleState(models.Model):
    """Latest lifecycle status for a document within a tenant workflow."""

    tenant_id = models.CharField(max_length=255)
    document_id = models.UUIDField()
    workflow_id = models.CharField(max_length=255, blank=True, default="")
    state = models.CharField(max_length=32)
    trace_id = models.CharField(max_length=255, blank=True, default="")
    run_id = models.CharField(max_length=255, blank=True, default="")
    ingestion_run_id = models.CharField(max_length=255, blank=True, default="")
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
            models.Index(
                fields=("tenant_id", "workflow_id"),
                name="doc_lifecycle_tenant_wf_idx",
            ),
        ]


class DocumentIngestionRun(models.Model):
    """Persistent metadata for the last ingestion run per tenant/case."""

    tenant_id = models.CharField(max_length=255)
    case = models.CharField(max_length=255)
    collection_id = models.CharField(max_length=255, blank=True, default="")
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
                fields=("tenant_id", "case"), name="document_ingestion_run_unique_case"
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant_id", "case"),
                name="doc_ing_run_tenant_case_idx",
            ),
            models.Index(
                fields=("tenant_id", "run_id"),
                name="doc_ing_run_tenant_run_idx",
            ),
        ]
