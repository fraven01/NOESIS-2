import factory
from factory.django import DjangoModelFactory, FileField

from documents.models import Document, DocumentType
from projects.tests.factories import ProjectFactory
from users.tests.factories import UserFactory


class DocumentTypeFactory(DjangoModelFactory):
    class Meta:
        model = DocumentType

    name = factory.Sequence(lambda n: f"Type {n}")
    description = factory.Faker("sentence")


class DocumentFactory(DjangoModelFactory):
    class Meta:
        model = Document

    file = FileField(filename="test.txt")
    type = factory.SubFactory(DocumentTypeFactory)
    project = factory.SubFactory(ProjectFactory)
    owner = factory.SubFactory(UserFactory)
