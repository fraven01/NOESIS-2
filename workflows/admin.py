from django.contrib import admin

from .models import WorkflowTemplate, WorkflowStep, WorkflowInstance


@admin.register(WorkflowTemplate)
class WorkflowTemplateAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`WorkflowTemplate`."""

    list_display = ("name", "created_at", "updated_at")


@admin.register(WorkflowStep)
class WorkflowStepAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`WorkflowStep`."""

    list_display = ("template", "order", "created_at", "updated_at")
    list_filter = ("template",)
    list_select_related = ("template",)


@admin.register(WorkflowInstance)
class WorkflowInstanceAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`WorkflowInstance`."""

    list_display = (
        "organization",
        "status",
        "created_at",
        "updated_at",
    )
    list_filter = ("organization", "status")
    list_select_related = ("organization",)
