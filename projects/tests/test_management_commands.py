import pytest
from django.core.management import call_command

from organizations.models import Organization
from projects.models import Project


class DummyProject:
    pk = 1
    slug = "dummy"
    organization = None

    def save(self, update_fields=None):
        pass


@pytest.mark.django_db
def test_assign_default_org_to_projects_without_org(monkeypatch):
    dummy = DummyProject()

    def base_filter(*args, **kwargs):
        return [dummy]

    def objects_filter(*args, **kwargs):
        raise AssertionError("should use _base_manager")

    monkeypatch.setattr(Project._base_manager, "filter", base_filter)
    monkeypatch.setattr(Project.objects, "filter", objects_filter)

    call_command("assign_default_org")

    assert isinstance(dummy.organization, Organization)
    assert dummy.organization.name == f"Legacy Org {dummy.slug}"
    assert dummy.organization.slug == f"legacy-org-{dummy.slug}"
