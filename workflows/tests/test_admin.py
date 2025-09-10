from django.contrib import admin

from workflows.admin import WorkflowInstanceAdmin
from workflows.models import WorkflowInstance


def test_workflowinstance_admin_displays_and_filters_organization():
    admin_instance = WorkflowInstanceAdmin(WorkflowInstance, admin.site)
    assert "organization" in admin_instance.list_display
    assert "project__organization" in admin_instance.list_filter
