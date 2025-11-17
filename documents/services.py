"""Document domain service helpers."""

from __future__ import annotations

from typing import Optional, Tuple
from uuid import UUID

from customers.models import Tenant

from cases.models import Case
from documents.contracts import NormalizedDocumentInputV1
from documents.models import DocumentCollection


def _resolve_case(tenant: Tenant, external_id: str | None) -> Case | None:
    if not external_id:
        return None
    candidate = (external_id or "").strip()
    if not candidate:
        return None
    try:
        return Case.objects.get(tenant=tenant, external_id=candidate)
    except Case.DoesNotExist:
        return None


def resolve_collection_for_document_input(
    input_contract: NormalizedDocumentInputV1,
    tenant: Tenant,
) -> Tuple[Optional[DocumentCollection], Optional[UUID]]:
    """Resolve logical and technical collections for *input_contract*."""

    document_collection: DocumentCollection | None = None
    technical_collection_id: UUID | None = input_contract.collection_id
    case = _resolve_case(tenant, input_contract.case_id)

    if input_contract.document_collection_id:
        try:
            document_collection = DocumentCollection.objects.select_related("case").get(
                tenant=tenant, id=input_contract.document_collection_id
            )
        except DocumentCollection.DoesNotExist as exc:  # pragma: no cover - guarded
            raise ValueError("document_collection_not_found") from exc
        if (
            technical_collection_id
            and technical_collection_id != document_collection.collection_id
        ):
            raise ValueError("document_collection_mismatch")
        technical_collection_id = document_collection.collection_id
        return document_collection, technical_collection_id

    if technical_collection_id is None:
        return None, None

    query = DocumentCollection.objects.select_related("case").filter(
        tenant=tenant, collection_id=technical_collection_id
    )
    if case is not None:
        document_collection = query.filter(case=case).first()
    if document_collection is None:
        document_collection = query.first()
    return document_collection, technical_collection_id
