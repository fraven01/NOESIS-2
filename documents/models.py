"""Database models supporting document lifecycle persistence."""

from __future__ import annotations

import uuid

from django.db import models
from django.utils import timezone


from . import framework_models  # noqa: F401


def _default_next_run_at() -> timezone.datetime:
    return timezone.now() + timezone.timedelta(hours=1)


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

    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_documents",
        db_index=True,
        help_text="User who created/uploaded this document (NULL for system)",
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="updated_documents",
        help_text="User who last modified this document",
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
    added_by_user = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    added_by_service_id = models.CharField(
        max_length=100,
        null=True,
        blank=True,
    )

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


class DocumentActivity(models.Model):
    """Audit trail for document access and modifications."""

    class ActivityType(models.TextChoices):
        VIEW = "VIEW", "Viewed"
        DOWNLOAD = "DOWNLOAD", "Downloaded"
        SEARCH = "SEARCH", "Found in Search"
        SHARE = "SHARE", "Shared"
        UPLOAD = "UPLOAD", "Uploaded"
        DELETE = "DELETE", "Deleted"
        TRANSFER = "TRANSFER", "Ownership Transferred"

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="activities",
        db_index=True,
    )
    user = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="document_activities",
    )

    activity_type = models.CharField(
        max_length=20,
        choices=ActivityType.choices,
        db_index=True,
    )

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=500, blank=True, default="")

    case_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        db_index=True,
    )
    trace_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        db_index=True,
    )

    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=("document", "-timestamp")),
            models.Index(fields=("user", "-timestamp")),
            models.Index(fields=("activity_type", "-timestamp")),
            models.Index(fields=("case_id", "-timestamp")),
        ]
        verbose_name = "Document Activity"
        verbose_name_plural = "Document Activities"

    def __str__(self) -> str:  # pragma: no cover - debug helper
        user_display = getattr(self.user, "id", None) or "system"
        return f"{self.activity_type} {self.document_id} by {user_display}"


class DocumentPermission(models.Model):
    """Document-level access control."""

    class PermissionType(models.TextChoices):
        VIEW = "VIEW", "View"
        DOWNLOAD = "DOWNLOAD", "Download"
        COMMENT = "COMMENT", "Comment"
        EDIT_META = "EDIT_META", "Edit Metadata"
        SHARE = "SHARE", "Share"
        DELETE = "DELETE", "Delete"

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="permissions",
    )
    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="document_permissions",
    )
    permission_type = models.CharField(
        max_length=20,
        choices=PermissionType.choices,
        db_index=True,
    )
    granted_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="granted_document_permissions",
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("document", "user", "permission_type"),
                name="document_permission_unique",
            )
        ]
        indexes = [
            models.Index(fields=("document", "user"), name="doc_perm_doc_user_idx"),
            models.Index(
                fields=("user", "permission_type"), name="doc_perm_user_type_idx"
            ),
        ]


class UserDocumentFavorite(models.Model):
    """User favorites for documents."""

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="document_favorites",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="favorites",
    )
    favorited_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("user", "document"),
                name="document_favorite_unique",
            )
        ]
        indexes = [
            models.Index(fields=("user", "favorited_at"), name="doc_fav_user_time_idx"),
            models.Index(fields=("document", "favorited_at"), name="doc_fav_doc_time_idx"),
        ]


class DocumentComment(models.Model):
    """Comments attached to documents (supports threading)."""

    class AnchorType(models.TextChoices):
        TEXT = "TEXT", "Text"
        PAGE = "PAGE", "Page"
        ASSET = "ASSET", "Asset"

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="comments",
    )
    user = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="document_comments",
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="replies",
    )
    text = models.TextField()
    anchor_type = models.CharField(
        max_length=20,
        choices=AnchorType.choices,
        null=True,
        blank=True,
    )
    anchor_reference = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=("document", "created_at"), name="doc_comment_doc_time_idx"),
            models.Index(fields=("user", "created_at"), name="doc_comment_user_time_idx"),
        ]


class DocumentMention(models.Model):
    """Mention references extracted from document comments."""

    comment = models.ForeignKey(
        DocumentComment,
        on_delete=models.CASCADE,
        related_name="mentions",
    )
    mentioned_user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="document_mentions",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("comment", "mentioned_user"),
                name="document_mention_unique",
            )
        ]
        indexes = [
            models.Index(fields=("mentioned_user", "created_at"), name="doc_mention_user_time_idx"),
        ]


class DocumentSubscription(models.Model):
    """User subscriptions for document-level notifications."""

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="document_subscriptions",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="subscriptions",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("user", "document"),
                name="document_subscription_unique",
            )
        ]
        indexes = [
            models.Index(fields=("user", "created_at"), name="doc_sub_user_time_idx"),
            models.Index(fields=("document", "created_at"), name="doc_sub_doc_time_idx"),
        ]


class DocumentNotification(models.Model):
    """In-app notifications for document events."""

    class EventType(models.TextChoices):
        MENTION = "MENTION", "Mention"
        COMMENT = "COMMENT", "Comment"
        FAVORITE = "FAVORITE", "Favorite"
        SAVED_SEARCH = "SAVED_SEARCH", "Saved Search"

    user = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="document_notifications",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="notifications",
    )
    comment = models.ForeignKey(
        DocumentComment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="notifications",
    )
    event_type = models.CharField(
        max_length=30,
        choices=EventType.choices,
        db_index=True,
    )
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=("user", "read_at"), name="doc_notif_user_read_idx"),
            models.Index(fields=("user", "created_at"), name="doc_notif_user_time_idx"),
            models.Index(fields=("event_type", "created_at"), name="doc_notif_type_time_idx"),
        ]


class NotificationEvent(models.Model):
    """External notification event queue."""

    class EventType(models.TextChoices):
        MENTION = "MENTION", "Mention"
        SAVED_SEARCH = "SAVED_SEARCH", "Saved Search"
        COMMENT_REPLY = "COMMENT_REPLY", "Comment Reply"

    user = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="notification_events",
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="notification_events",
    )
    comment = models.ForeignKey(
        DocumentComment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="notification_events",
    )
    event_type = models.CharField(
        max_length=30,
        choices=EventType.choices,
        db_index=True,
    )
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=("user", "created_at"), name="notif_evt_user_time_idx"),
            models.Index(
                fields=("event_type", "created_at"), name="notif_evt_type_time_idx"
            ),
        ]


class NotificationDelivery(models.Model):
    """External notification delivery tracking."""

    class Channel(models.TextChoices):
        EMAIL = "EMAIL", "Email"

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        SENT = "SENT", "Sent"
        FAILED = "FAILED", "Failed"
        SKIPPED = "SKIPPED", "Skipped"

    event = models.ForeignKey(
        NotificationEvent,
        on_delete=models.CASCADE,
        related_name="deliveries",
    )
    channel = models.CharField(
        max_length=20,
        choices=Channel.choices,
        db_index=True,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    attempts = models.IntegerField(default=0)
    next_attempt_at = models.DateTimeField(default=timezone.now, db_index=True)
    last_error = models.TextField(blank=True, default="")
    sent_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("event", "channel"),
                name="notification_delivery_unique",
            )
        ]
        indexes = [
            models.Index(
                fields=("status", "next_attempt_at"),
                name="notif_delivery_due_idx",
            ),
        ]


class SavedSearch(models.Model):
    """User-defined saved searches with scheduled alerts."""

    class AlertFrequency(models.TextChoices):
        HOURLY = "HOURLY", "Hourly"
        DAILY = "DAILY", "Daily"
        WEEKLY = "WEEKLY", "Weekly"

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="saved_searches",
    )
    name = models.CharField(max_length=255)
    query = models.TextField(blank=True, default="")
    filters = models.JSONField(default=dict, blank=True)
    enable_alerts = models.BooleanField(default=True)
    alert_frequency = models.CharField(
        max_length=20,
        choices=AlertFrequency.choices,
        default=AlertFrequency.HOURLY,
    )
    last_run_at = models.DateTimeField(null=True, blank=True)
    next_run_at = models.DateTimeField(
        default=_default_next_run_at,
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=("user", "created_at"), name="saved_search_user_time_idx"),
            models.Index(fields=("enable_alerts", "next_run_at"), name="saved_search_due_idx"),
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
