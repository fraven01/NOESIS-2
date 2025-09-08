from django.contrib import admin

from .models import Project, WorkflowInstance


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`Project`."""

    list_display = ("name", "status", "owner", "created_at", "updated_at")
    list_filter = ("status",)


@admin.register(WorkflowInstance)
class WorkflowInstanceAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`WorkflowInstance`."""

    list_display = ("project", "template", "created_at", "updated_at")
