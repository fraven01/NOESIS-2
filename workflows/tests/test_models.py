import pytest

from .factories import (
    WorkflowInstanceFactory,
    WorkflowStepFactory,
    WorkflowTemplateFactory,
)


@pytest.mark.django_db
def test_workflow_template_factory():
    template = WorkflowTemplateFactory()
    assert template.pk is not None


@pytest.mark.django_db
def test_workflow_step_factory_creates_step():
    step = WorkflowStepFactory()
    assert step.template.steps.count() == 1


@pytest.mark.django_db
def test_workflow_instance_factory_creates_instance():
    instance = WorkflowInstanceFactory()
    assert instance.organization is not None
    assert instance.status == instance.STATUS_DRAFT
