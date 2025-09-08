from django.contrib import admin

from .models import Document, DocumentType


@admin.register(DocumentType)
class DocumentTypeAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`DocumentType`."""

    list_display = ("name", "workflow", "created_at", "updated_at")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`Document`."""

    list_display = ("type", "owner", "status", "created_at", "updated_at")
    list_filter = ("status", "type")

