import factory
from factory.django import DjangoModelFactory

from workflows.models import WorkflowTemplate, WorkflowStep


class WorkflowTemplateFactory(DjangoModelFactory):
    class Meta:
        model = WorkflowTemplate

    name = factory.Sequence(lambda n: f"Template {n}")


class WorkflowStepFactory(DjangoModelFactory):
    class Meta:
        model = WorkflowStep

    template = factory.SubFactory(WorkflowTemplateFactory)
    order = factory.Sequence(lambda n: n)
    instructions = factory.Faker("sentence")
