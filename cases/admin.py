"""Django admin configuration for cases app."""

from django.contrib import admin

from cases.models import Case, CaseEvent, CaseMembership


class CaseMembershipInline(admin.TabularInline):
    """Inline admin for managing case memberships."""

    model = CaseMembership
    extra = 1
    autocomplete_fields = ["user", "granted_by"]
    readonly_fields = ["created_at"]
    fields = ["user", "granted_by", "created_at"]


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    """Admin interface for Case model."""

    list_display = ["external_id", "title", "status", "phase", "created_at"]
    list_filter = ["status", "phase", "created_at"]
    search_fields = ["external_id", "title"]
    readonly_fields = ["id", "created_at", "updated_at", "closed_at"]
    inlines = [CaseMembershipInline]

    fieldsets = (
        (
            None,
            {
                "fields": (
                    "id",
                    "tenant",
                    "external_id",
                    "title",
                    "status",
                    "phase",
                )
            },
        ),
        (
            "Metadata",
            {
                "fields": ("metadata",),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at", "closed_at"),
                "classes": ("collapse",),
            },
        ),
    )


@admin.register(CaseEvent)
class CaseEventAdmin(admin.ModelAdmin):
    """Admin interface for CaseEvent model."""

    list_display = [
        "case",
        "event_type",
        "source",
        "graph_name",
        "workflow_id",
        "created_at",
    ]
    list_filter = ["event_type", "source", "graph_name", "created_at"]
    search_fields = ["case__external_id", "event_type", "workflow_id", "trace_id"]
    readonly_fields = [
        "case",
        "tenant",
        "event_type",
        "source",
        "graph_name",
        "ingestion_run",
        "workflow_id",
        "collection_id",
        "trace_id",
        "payload",
        "created_at",
    ]

    def has_add_permission(self, request):
        """Case events are system-generated only."""
        return False

    def has_change_permission(self, request, obj=None):
        """Case events are immutable."""
        return False
