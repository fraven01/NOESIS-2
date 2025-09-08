import factory
from factory.django import DjangoModelFactory

from projects.models import Project, WorkflowInstance
from users.tests.factories import UserFactory
from workflows.tests.factories import WorkflowTemplateFactory


class ProjectFactory(DjangoModelFactory):
    class Meta:
        model = Project

    name = factory.Sequence(lambda n: f"Project {n}")
    description = factory.Faker("sentence")
    owner = factory.SubFactory(UserFactory)


class WorkflowInstanceFactory(DjangoModelFactory):
    class Meta:
        model = WorkflowInstance

    project = factory.SubFactory(ProjectFactory)
    template = factory.SubFactory(WorkflowTemplateFactory)
    state = factory.LazyFunction(dict)
