import pytest

from organizations.tests.factories import OrganizationFactory
from organizations.utils import set_current_organization
from workflows.tests.factories import WorkflowInstanceFactory
from workflows.models import WorkflowInstance


@pytest.fixture
def workflow_instance_factory():
    return WorkflowInstanceFactory


@pytest.mark.django_db
def test_workflowinstance_missing_org_returns_empty(workflow_instance_factory):
    org = OrganizationFactory()
    workflow_instance_factory(organization=org)
    assert list(WorkflowInstance.objects.all()) == []


@pytest.mark.django_db
def test_workflowinstance_wrong_org_returns_empty(workflow_instance_factory):
    org1 = OrganizationFactory()
    org2 = OrganizationFactory()
    workflow_instance_factory(organization=org1)
    with set_current_organization(org2):
        assert list(WorkflowInstance.objects.all()) == []


@pytest.mark.django_db
def test_workflowinstance_correct_org_returns_result(workflow_instance_factory):
    org = OrganizationFactory()
    instance = workflow_instance_factory(organization=org)
    with set_current_organization(org):
        assert list(WorkflowInstance.objects.all()) == [instance]
