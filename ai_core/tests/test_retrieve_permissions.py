from __future__ import annotations

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.nodes import retrieve
from customers.models import Tenant
from documents.models import Document, DocumentPermission
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def _tool_context(*, tenant: Tenant, user_id: str):
    scope = ScopeContext(
        tenant_id=tenant.schema_name,
        trace_id="trace-1",
        invocation_id="inv-1",
        run_id="run-1",
        user_id=str(user_id),
    )
    return scope.to_tool_context(business=BusinessContext())


def test_filter_matches_by_permissions_allows_granted_docs(
    test_tenant_schema_name,
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = UserFactory()

    allowed_doc = Document.objects.create(
        tenant=tenant, hash="perm-hash-a", source="upload", metadata={}
    )
    denied_doc = Document.objects.create(
        tenant=tenant, hash="perm-hash-b", source="upload", metadata={}
    )

    DocumentPermission.objects.create(
        document=allowed_doc,
        user=user,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    context = _tool_context(tenant=tenant, user_id=str(user.id))
    matches = [
        {"id": str(allowed_doc.id), "score": 0.9},
        {"id": str(denied_doc.id), "score": 0.8},
    ]

    filtered = retrieve._filter_matches_by_permissions(
        matches,
        context=context,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    assert [match["id"] for match in filtered] == [str(allowed_doc.id)]
