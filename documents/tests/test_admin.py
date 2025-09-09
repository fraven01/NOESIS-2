from django.contrib import admin

from documents.admin import DocumentAdmin
from documents.models import Document


def test_document_admin_displays_and_filters_organization():
    admin_instance = DocumentAdmin(Document, admin.site)
    assert "organization" in admin_instance.list_display
    assert "project__organization" in admin_instance.list_filter
