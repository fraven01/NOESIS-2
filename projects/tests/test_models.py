import pytest

from .factories import ProjectFactory, WorkflowInstanceFactory


@pytest.mark.django_db
def test_project_factory_creates_project():
    project = ProjectFactory()
    assert project.pk is not None
    assert project.organization is not None


@pytest.mark.django_db
def test_workflow_instance_factory_creates_instance():
    instance = WorkflowInstanceFactory()
    assert instance.project.workflow == instance
