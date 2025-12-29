from django.contrib import admin
from .models import (
    Document,
    DocumentCollection,
    DocumentAsset,
    DocumentIngestionRun,
    DocumentCollectionMembership,
    DocumentActivity,
    DocumentPermission,
    UserDocumentFavorite,
    DocumentComment,
    DocumentMention,
    DocumentNotification,
    SavedSearch,
)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "tenant",
        "source",
        "lifecycle_state",
        "created_by",
        "updated_by",
        "created_at",
    )
    list_filter = ("lifecycle_state", "tenant", "created_by")
    search_fields = ("id", "source", "hash")
    readonly_fields = ("created_at", "updated_at")


@admin.register(DocumentCollection)
class DocumentCollectionAdmin(admin.ModelAdmin):
    list_display = ("name", "key", "tenant", "collection_id", "type")
    list_filter = ("type", "visibility", "tenant")
    search_fields = ("name", "key", "collection_id")


@admin.register(DocumentAsset)
class DocumentAssetAdmin(admin.ModelAdmin):
    list_display = (
        "asset_id",
        "tenant",
        "document",
        "workflow_id",
        "media_type",
        "created_at",
    )
    list_filter = ("media_type", "tenant", "workflow_id")
    search_fields = ("asset_id", "document__id", "workflow_id")


@admin.register(DocumentIngestionRun)
class DocumentIngestionRunAdmin(admin.ModelAdmin):
    list_display = ("run_id", "tenant_id", "status", "queued_at", "finished_at")
    list_filter = ("status", "tenant_id")
    search_fields = ("run_id", "trace_id")


@admin.register(DocumentCollectionMembership)
class DocumentCollectionMembershipAdmin(admin.ModelAdmin):
    list_display = (
        "document",
        "collection",
        "added_at",
        "added_by_user",
        "added_by_service_id",
    )
    list_filter = ("added_by_user", "added_by_service_id", "collection__tenant")


@admin.register(DocumentActivity)
class DocumentActivityAdmin(admin.ModelAdmin):
    list_display = (
        "document",
        "user",
        "activity_type",
        "timestamp",
        "case_id",
    )
    list_filter = ("activity_type", "user")
    search_fields = ("document__id", "case_id", "trace_id")


@admin.register(DocumentPermission)
class DocumentPermissionAdmin(admin.ModelAdmin):
    list_display = ("document", "user", "permission_type", "granted_at", "expires_at")
    list_filter = ("permission_type", "user")
    search_fields = ("document__id",)


@admin.register(UserDocumentFavorite)
class UserDocumentFavoriteAdmin(admin.ModelAdmin):
    list_display = ("document", "user", "favorited_at")
    list_filter = ("user",)
    search_fields = ("document__id",)


@admin.register(DocumentComment)
class DocumentCommentAdmin(admin.ModelAdmin):
    list_display = ("document", "user", "parent", "created_at")
    list_filter = ("document", "user")
    search_fields = ("document__id", "text")


@admin.register(DocumentMention)
class DocumentMentionAdmin(admin.ModelAdmin):
    list_display = ("comment", "mentioned_user", "created_at")
    list_filter = ("mentioned_user",)
    search_fields = ("comment__id", "mentioned_user__username")


@admin.register(DocumentNotification)
class DocumentNotificationAdmin(admin.ModelAdmin):
    list_display = ("user", "event_type", "document", "created_at", "read_at")
    list_filter = ("event_type", "user")
    search_fields = ("document__id",)


@admin.register(SavedSearch)
class SavedSearchAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "enable_alerts", "alert_frequency", "next_run_at")
    list_filter = ("enable_alerts", "alert_frequency")
    search_fields = ("name", "user__username")
