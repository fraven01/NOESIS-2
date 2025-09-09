from django.contrib import admin

from projects.admin import ProjectAdmin, WorkflowInstanceAdmin
from projects.models import Project, WorkflowInstance


def test_project_admin_displays_and_filters_organization():
    admin_instance = ProjectAdmin(Project, admin.site)
    assert "organization" in admin_instance.list_display
    assert "organization" in admin_instance.list_filter


def test_workflowinstance_admin_displays_and_filters_organization():
    admin_instance = WorkflowInstanceAdmin(WorkflowInstance, admin.site)
    assert "organization" in admin_instance.list_display
    assert "project__organization" in admin_instance.list_filter
