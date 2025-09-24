import pytest
from django.core.management import call_command
from django_tenants.utils import schema_context

from documents.models import Document
from organizations.models import Organization
from organizations.utils import set_current_organization


def _wipe_demo_seed():
    call_command("create_demo_data", "--wipe", "--include-org")


@pytest.mark.django_db
def test_create_demo_data_chaos_sets_error_detail(monkeypatch):
    _wipe_demo_seed()

    storage = Document._meta.get_field("file").storage
    original_save = storage.save

    def flaky_save(name, content, max_length=None):
        if not getattr(flaky_save, "_fired", False):
            flaky_save._fired = True
            raise OSError("simulated save failure")
        return original_save(name, content, max_length=max_length)

    flaky_save._fired = False
    monkeypatch.setattr(storage, "save", flaky_save)

    call_command("create_demo_data", "--profile", "chaos", "--seed", "1337")

    try:
        with schema_context("demo"):
            org = Organization.objects.get(slug="demo")
            with set_current_organization(org):
                documents = list(Document.objects.filter(project__organization=org))
    finally:
        _wipe_demo_seed()

    invalid_found = False
    error_detail_found = False
    for document in documents:
        meta = document.meta if isinstance(document.meta, dict) else {}
        if meta.get("invalid") is True:
            invalid_found = True
        issues = meta.get("issues", [])
        if any(isinstance(issue, dict) and "error_detail" in issue for issue in issues):
            error_detail_found = True

    assert invalid_found, "Expected at least one invalid document in chaos profile"
    assert (
        error_detail_found
    ), "Expected at least one document to include an error_detail entry in metadata"
