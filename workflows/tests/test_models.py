import pytest

from .factories import WorkflowStepFactory, WorkflowTemplateFactory


@pytest.mark.django_db
def test_workflow_template_factory():
    template = WorkflowTemplateFactory()
    assert template.pk is not None


@pytest.mark.django_db
def test_workflow_step_factory_creates_step():
    step = WorkflowStepFactory()
    assert step.template.steps.count() == 1
