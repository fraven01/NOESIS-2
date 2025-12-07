from django.contrib import admin
from .models import (
    Document,
    DocumentCollection,
    DocumentLifecycleState,
    DocumentAsset,
    DocumentIngestionRun,
    DocumentCollectionMembership,
)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "tenant", "source", "lifecycle_state", "created_at")
    list_filter = ("lifecycle_state", "tenant")
    search_fields = ("id", "source", "hash")
    readonly_fields = ("created_at", "updated_at")


@admin.register(DocumentCollection)
class DocumentCollectionAdmin(admin.ModelAdmin):
    list_display = ("name", "key", "tenant", "collection_id", "type")
    list_filter = ("type", "visibility", "tenant")
    search_fields = ("name", "key", "collection_id")


@admin.register(DocumentLifecycleState)
class DocumentLifecycleStateAdmin(admin.ModelAdmin):
    list_display = ("document_id", "tenant_id", "workflow_id", "state", "changed_at")
    list_filter = ("state", "tenant_id", "workflow_id")
    search_fields = ("document_id", "run_id")


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
    list_display = ("document", "collection", "added_at", "added_by")
    list_filter = ("added_by", "collection__tenant")
