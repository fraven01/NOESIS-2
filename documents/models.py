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
    embedding_profile = models.CharField(max_length=255, blank=True, default="")
    soft_deleted_at = models.DateTimeField(null=True, blank=True)
    documents = models.ManyToManyField(
        "Document",
        through="DocumentCollectionMembership",
        related_name="collections",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "key"),
                name="document_collection_unique_key",
            ),
            models.UniqueConstraint(
                fields=("tenant", "collection_id"),
                name="unique_collection_id_per_tenant",
            ),
        ]
        indexes = [
            models.Index(
                fields=("tenant", "case"),
                name="doc_collection_tenant_case_idx",
            ),
            models.Index(
                fields=("tenant", "embedding_profile"),
                name="doc_coll_tenant_profile_idx",
            ),
        ]

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.name} ({self.key})"


class Document(models.Model):
    """Persisted document metadata shared across collections.

    Context fields (workflow_id, trace_id, case_id) enable traceability
    and support future tenant-specific workflow configurations.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="documents",
    )
    hash = models.CharField(max_length=128)
    source = models.CharField(max_length=255)
    external_id = models.CharField(max_length=255, blank=True, null=True)
    metadata = models.JSONField(blank=True, default=dict)

    # Context fields for traceability and tenant-specific workflows
    workflow_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        default="",
        db_index=True,
        help_text="Workflow that created/modified this document",
    )
    trace_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        default="",
        help_text="Trace ID for end-to-end correlation",
    )
    case_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        db_index=True,
        help_text="Business case this document belongs to",
    )

    lifecycle_state = models.CharField(
        max_length=32,
        default="pending",
        db_index=True,
    )
    lifecycle_updated_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    soft_deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "source", "hash"),
                name="document_unique_source_hash",
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant", "source"), name="document_tenant_source_idx"
            ),
            models.Index(fields=("tenant", "hash"), name="document_tenant_hash_idx"),
            models.Index(
                fields=("tenant", "lifecycle_state"),
                name="doc_tenant_state_idx",
            ),
            models.Index(
                fields=("tenant", "case_id"),
                name="doc_tenant_case_idx",
            ),
        ]


class DocumentCollectionMembership(models.Model):
    """Membership relation between documents and logical collections."""

    id = models.BigAutoField(primary_key=True)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="collection_memberships",
    )
    collection = models.ForeignKey(
        DocumentCollection,
        on_delete=models.CASCADE,
        related_name="collection_memberships",
    )
    added_at = models.DateTimeField(auto_now_add=True)
    added_by = models.CharField(max_length=255, default="system")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("document", "collection"),
                name="document_collection_membership_unique",
            )
        ]
        indexes = [
            models.Index(fields=("collection",), name="document_collection_idx"),
            models.Index(fields=("document",), name="collection_document_idx"),
        ]


class DocumentIngestionRun(models.Model):
    """Persistent metadata for the last ingestion run per tenant/case."""

    tenant_id = models.CharField(max_length=255)
    case = models.CharField(max_length=255, null=True, blank=True)
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


class DocumentAsset(models.Model):
    """
    Persistence for non-document assets (chunks, images, etc.) associated with a document.
    These are logical assets that may or may not map 1:1 to a file blob.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="document_assets",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="assets",
    )
    workflow_id = models.CharField(max_length=255)
    asset_id = models.UUIDField()
    collection_id = models.UUIDField(null=True, blank=True)

    media_type = models.CharField(max_length=255)
    blob_metadata = models.JSONField(blank=True, default=dict)

    # Optional content cache (e.g. for text chunks)
    content = models.TextField(blank=True, null=True)

    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "asset_id", "workflow_id"),
                name="document_asset_unique_identity",
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant", "document"),
                name="doc_asset_tenant_doc_idx",
            ),
            models.Index(
                fields=("tenant", "asset_id"),
                name="doc_asset_tenant_asset_idx",
            ),
        ]
