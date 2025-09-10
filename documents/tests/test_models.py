import pytest

from .factories import DocumentFactory, DocumentTypeFactory


@pytest.mark.django_db
def test_document_type_factory():
    doc_type = DocumentTypeFactory()
    assert doc_type.pk is not None


@pytest.mark.django_db
def test_document_factory_creates_document():
    document = DocumentFactory()
    assert document.project is not None
    assert document.owner is not None
    assert document.title != ""
