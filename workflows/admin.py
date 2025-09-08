from django.contrib import admin

from .models import Workflow, WorkflowStep


@admin.register(Workflow)
class WorkflowAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`Workflow`."""

    list_display = ("name", "created_at", "updated_at")


@admin.register(WorkflowStep)
class WorkflowStepAdmin(admin.ModelAdmin):
    """Admin configuration for :class:`WorkflowStep`."""

    list_display = ("workflow", "order", "created_at", "updated_at")
    list_filter = ("workflow",)

