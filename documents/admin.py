from django.contrib import admin

from .models import Document, DocumentType


@admin.register(DocumentType)
class DocumentTypeAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`DocumentType`."""

    list_display = ("name", "created_at", "updated_at")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`Document`."""

    list_display = (
        "title",
        "type",
        "project",
        "organization",
        "owner",
        "status",
        "created_at",
        "updated_at",
    )
    list_filter = ("status", "type", "project__organization", "project")

    @staticmethod
    def organization(obj):
        return obj.project.organization
