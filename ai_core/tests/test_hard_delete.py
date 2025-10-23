import uuid

import pytest
from django.core.exceptions import PermissionDenied

from ai_core.rag import vector_client
from ai_core.rag.hard_delete import hard_delete
from ai_core.rag.schemas import Chunk
from organizations.models import Organization, OrgMembership
from profiles.models import UserProfile
from users.models import User


def _insert_document(tenant_id: str, embedding_dim: int = 1536) -> str:
    vector_client.reset_default_client()
    chunk = Chunk(
        content="Doc A",
        meta={
            "tenant": tenant_id,
            "hash": "hash-doc-a",
            "external_id": "doc-a",
            "source": "tests",
        },
        embedding=[float(i) / embedding_dim for i in range(1, embedding_dim + 1)],
    )
    client = vector_client.get_default_client()
    client.upsert_chunks([chunk])
    with client.connection() as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE tenant_id = %s",
                (uuid.UUID(tenant_id),),
            )
            row = cur.fetchone()
    assert row is not None, "document insertion failed"
    return str(row[0])


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_service_key(monkeypatch, settings):
    settings.RAG_INTERNAL_KEYS = ["service-key"]
    tenant_id = str(uuid.uuid4())

    document_id = _insert_document(tenant_id)

    spans: list[dict[str, object]] = []

    def _capture_span(
        name: str,
        *,
        attributes: dict[str, object] | None = None,
        trace_id: str | None = None,
    ) -> None:
        spans.append(
            {"trace_id": trace_id, "node_name": name, "metadata": attributes or {}}
        )

    monkeypatch.setattr("ai_core.rag.hard_delete.record_span", _capture_span)

    result = hard_delete(
        tenant_id,
        [document_id],
        "cleanup",
        "TCK-1",
        actor={"internal_key": "service-key", "label": "ops-bot"},
        trace_id="trace-123",
    )

    assert result["documents_deleted"] == 1
    assert result["deleted_ids"] == [document_id]
    assert result["not_found"] == 0
    assert result["visibility"] == "deleted"

    repeat = hard_delete(
        tenant_id,
        [document_id],
        "cleanup",
        "TCK-1",
        actor={"internal_key": "service-key"},
        trace_id="trace-124",
    )

    assert repeat["documents_deleted"] == 0
    assert repeat["not_found"] == 1

    assert len(spans) == 2, "expected Langfuse spans for each invocation"
    assert spans[0]["node_name"] == "rag.hard_delete"
    assert spans[0]["trace_id"] == "trace-123"
    assert spans[0]["metadata"]["documents_deleted"] == 1
    assert spans[1]["trace_id"] == "trace-124"

    with vector_client.get_default_client().connection() as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE tenant_id = %s",
                (uuid.UUID(tenant_id),),
            )
            remaining = cur.fetchone()[0]
    assert remaining == 0


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_service_key_with_scoped_session(monkeypatch, settings):
    settings.RAG_INTERNAL_KEYS = ["service-key"]
    tenant_id = str(uuid.uuid4())

    document_id = _insert_document(tenant_id)

    trace_id = "trace-321"
    case_id = "case-123"
    session_salt = "||".join((trace_id, case_id, tenant_id))

    spans: list[dict[str, object]] = []

    def _capture_span(
        trace_id: str, node_name: str, metadata: dict[str, object]
    ) -> None:
        spans.append(
            {"trace_id": trace_id, "node_name": node_name, "metadata": metadata}
        )

    monkeypatch.setattr("ai_core.rag.hard_delete.record_span", _capture_span)

    result = hard_delete(
        tenant_id,
        [document_id],
        "cleanup",
        "TCK-5",
        actor={"internal_key": "service-key"},
        trace_id=trace_id,
        session_salt=session_salt,
        session_scope=(tenant_id, case_id, session_salt),
    )

    assert result["documents_deleted"] == 1
    assert spans, "expected span emission"
    assert spans[0]["metadata"].get("session_salt") == session_salt


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_requires_authorisation(monkeypatch, settings):
    settings.RAG_INTERNAL_KEYS = ["service-key"]
    tenant_id = str(uuid.uuid4())

    document_id = _insert_document(tenant_id)

    with pytest.raises(PermissionDenied):
        hard_delete(tenant_id, [document_id], "cleanup", "TCK-2")


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_allows_admin_user(monkeypatch, settings):
    tenant_id = str(uuid.uuid4())
    document_id = _insert_document(tenant_id)

    user = User.objects.create_user(username="admin", email="admin@example.com")
    UserProfile.objects.update_or_create(
        user=user,
        defaults={"role": UserProfile.Roles.ADMIN, "is_active": True},
    )

    result = hard_delete(
        tenant_id,
        [document_id],
        "manual",
        "TCK-3",
        actor={"user_id": user.pk},
    )

    assert result["documents_deleted"] == 1


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_allows_org_admin(monkeypatch, settings):
    tenant_id = str(uuid.uuid4())
    document_id = _insert_document(tenant_id)

    user = User.objects.create_user(username="org-admin", email="org@example.com")
    UserProfile.objects.update_or_create(user=user, defaults={"is_active": True})
    organization = Organization.objects.create(name="Org", slug="org")
    OrgMembership.objects.create(
        organization=organization,
        user=user,
        role=OrgMembership.Role.ADMIN,
    )

    result = hard_delete(
        tenant_id,
        [document_id],
        "manual",
        "TCK-4",
        actor={"user_id": user.pk, "organization_id": str(organization.id)},
    )

    assert result["documents_deleted"] == 1
