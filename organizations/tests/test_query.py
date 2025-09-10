import pytest

from organizations.tests.factories import OrganizationFactory
from projects.tests.factories import ProjectFactory
from workflows.tests.factories import WorkflowInstanceFactory
from documents.tests.factories import DocumentFactory
from organizations.utils import set_current_organization
from projects.models import Project
from workflows.models import WorkflowInstance
from documents.models import Document


@pytest.mark.django_db
def test_project_missing_org_returns_empty():
    org = OrganizationFactory()
    ProjectFactory(organization=org)
    assert list(Project.objects.all()) == []


@pytest.mark.django_db
def test_project_wrong_org_returns_empty():
    org1 = OrganizationFactory()
    org2 = OrganizationFactory()
    ProjectFactory(organization=org1)
    with set_current_organization(org2):
        assert list(Project.objects.all()) == []


@pytest.mark.django_db
def test_project_correct_org_returns_result():
    org = OrganizationFactory()
    project = ProjectFactory(organization=org)
    with set_current_organization(org):
        assert list(Project.objects.all()) == [project]


@pytest.mark.django_db
def test_document_missing_org_returns_empty():
    org = OrganizationFactory()
    DocumentFactory(project__organization=org)
    assert list(Document.objects.all()) == []


@pytest.mark.django_db
def test_document_wrong_org_returns_empty():
    org1 = OrganizationFactory()
    org2 = OrganizationFactory()
    DocumentFactory(project__organization=org1)
    with set_current_organization(org2):
        assert list(Document.objects.all()) == []


@pytest.mark.django_db
def test_document_correct_org_returns_result():
    org = OrganizationFactory()
    document = DocumentFactory(project__organization=org)
    with set_current_organization(org):
        assert list(Document.objects.all()) == [document]


@pytest.mark.django_db
def test_workflowinstance_missing_org_returns_empty():
    org = OrganizationFactory()
    WorkflowInstanceFactory(project__organization=org)
    assert list(WorkflowInstance.objects.all()) == []


@pytest.mark.django_db
def test_workflowinstance_wrong_org_returns_empty():
    org1 = OrganizationFactory()
    org2 = OrganizationFactory()
    WorkflowInstanceFactory(project__organization=org1)
    with set_current_organization(org2):
        assert list(WorkflowInstance.objects.all()) == []


@pytest.mark.django_db
def test_workflowinstance_correct_org_returns_result():
    org = OrganizationFactory()
    instance = WorkflowInstanceFactory(project__organization=org)
    with set_current_organization(org):
        assert list(WorkflowInstance.objects.all()) == [instance]
