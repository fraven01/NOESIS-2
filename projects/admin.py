from django.contrib import admin

from .models import Project


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`Project`."""

    list_display = (
        "name",
        "organization",
        "status",
        "owner",
        "created_at",
        "updated_at",
    )
    list_filter = ("organization", "status")
    list_select_related = ("organization", "owner")
