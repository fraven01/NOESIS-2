"""Global test fixtures for tests/* subtree."""

from uuid import uuid4

import pytest
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
    monkeypatch.setattr(repo, "DEFAULT_LIFECYCLE_STORE", store, raising=False)
    yield


@pytest.fixture(autouse=True)
def cleanup_tenants_after_test(request, django_db_blocker, tenant_pool_schemas):
    """
    Ensure tenant cleanup runs for tests under tests/ subtree.

    Some teardown paths rely on conftest.cleanup_test_tenants; mirror the root
    fixture here so AttributeErrors go away and schemas are dropped.
    """
    yield
    if request.node.get_closest_marker("slow") or request.node.get_closest_marker(
        "tenant_ops"
    ):
        return
    with django_db_blocker.unblock():
        from django.db import connection

        try:
            cleanup_test_tenants(preserve=list(tenant_pool_schemas.values()))
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass


@pytest.fixture
def tenant(request, tenant_pool, django_db_blocker):
    """Provide a pooled tenant for tests in tests/ subtree."""
    if "transactional_db" in request.fixturenames:
        from testsupport.tenant_fixtures import create_test_tenant

        with django_db_blocker.unblock():
            return create_test_tenant(
                schema_name=f"txn_{uuid4().hex[:8]}",
                name="Transactional Tenant",
            )
    return tenant_pool["alpha"]


@pytest.fixture(autouse=True)
def force_inmemory_repository(monkeypatch):
    """
    Ensure tests in this subtree use the in-memory repository (avoid FK to tenants).
    """
    from documents.repository import InMemoryDocumentsRepository
    import ai_core.services as services

    # Reset cached repo and force in-memory
    monkeypatch.setattr(
        services, "_DOCUMENTS_REPOSITORY", InMemoryDocumentsRepository(), raising=False
    )
    # monkeypatch.setenv("DOCUMENTS_REPOSITORY_CLASS", "documents.repository.InMemoryDocumentsRepository")
    yield
