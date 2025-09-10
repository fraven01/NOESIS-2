from django.contrib import admin

from projects.admin import ProjectAdmin
from projects.models import Project


def test_project_admin_displays_and_filters_organization():
    admin_instance = ProjectAdmin(Project, admin.site)
    assert "organization" in admin_instance.list_display
    assert "organization" in admin_instance.list_filter
