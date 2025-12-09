"""Global test fixtures for tests/* subtree."""

import pytest
from django.conf import settings
from testsupport.tenant_fixtures import cleanup_test_tenants


@pytest.fixture(autouse=True)
def use_inmemory_lifecycle_store(monkeypatch):
    """
    Force the in-memory lifecycle store in tests to avoid FK issues against the DB.

    PersistentDocumentLifecycleStore now enforces a Tenant FK; many tests rely on
    string-only tenant IDs. By overriding the default lifecycle store to the
    in-memory variant we keep those tests isolated from the tenant table.
    """
    from documents.repository import DocumentLifecycleStore
    import documents.repository as repo

    store = DocumentLifecycleStore()
    monkeypatch.setattr(
        settings,
        "DOCUMENT_LIFECYCLE_STORE_CLASS",
        "documents.repository.DocumentLifecycleStore",
        raising=False,
    )
    monkeypatch.setattr(repo, "DEFAULT_LIFECYCLE_STORE", store, raising=False)
    yield


@pytest.fixture(autouse=True)
def cleanup_tenants_after_test(django_db_blocker):
    """
    Ensure tenant cleanup runs for tests under tests/ subtree.

    Some teardown paths rely on conftest.cleanup_test_tenants; mirror the root
    fixture here so AttributeErrors go away and schemas are dropped.
    """
    yield
    with django_db_blocker.unblock():
        cleanup_test_tenants()
