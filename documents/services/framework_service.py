"""Domain service for framework agreement profiles."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from django.db import transaction

from customers.models import Tenant
from documents.framework_models import FrameworkDocument, FrameworkProfile
from documents.models import DocumentCollection

logger = logging.getLogger(__name__)


def persist_profile(
    *,
    tenant_schema: str,
    gremium_identifier: str,
    gremium_name_raw: str,
    agreement_type: str,
    structure: Dict[str, Any],
    document_collection_id: str,
    document_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    force_reanalysis: bool = False,
    audit_meta: Dict[str, Any] | None = None,
    analysis_metadata: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    completeness_score: float | None = None,
    missing_components: list[str] | None = None,
) -> FrameworkProfile:
    """
    Persist a FrameworkProfile and link the main document.

    Handles versioning: checks for existing current profile for the gremium.
    If exists and force_reanalysis=False, raises ValueError.
    If exists and force_reanalysis=True, sets old to not current and creates new version.

    Args:
        audit_meta: Pre-MVP ID Contract audit metadata for traceability.
            Contains trace_id, invocation_id, created_by_user_id, last_hop_service_id.
            Stored in FrameworkProfile.audit_meta for entity tracking.
    """
    if analysis_metadata is None:
        analysis_metadata = {}
    if metadata is None:
        metadata = {}

    # Get tenant object
    try:
        tenant = Tenant.objects.get(schema_name=tenant_schema)
    except Tenant.DoesNotExist:
        raise ValueError(f"Tenant not found: {tenant_schema}")

    # Check for existing profile
    with transaction.atomic():
        existing_profile = FrameworkProfile.objects.filter(
            tenant=tenant,
            gremium_identifier=gremium_identifier,
            is_current=True,
        ).first()

        if existing_profile and not force_reanalysis:
            # Profile already exists and not forcing reanalysis
            raise ValueError(
                f"Framework profile already exists for {gremium_identifier}. "
                f"Use force_reanalysis=true to create new version."
            )

        # Determine version
        if existing_profile:
            # Set old profile to not current and increment version
            existing_profile.is_current = False
            existing_profile.save(update_fields=["is_current", "updated_at"])
            version = existing_profile.version + 1
        else:
            version = 1

        # Get document collection
        try:
            doc_collection = DocumentCollection.objects.get(
                id=UUID(document_collection_id),
                tenant=tenant,
            )
        except DocumentCollection.DoesNotExist:
            raise ValueError(f"DocumentCollection not found: {document_collection_id}")

        # Generate name from gremium identifier and type
        name = f"{agreement_type.upper()} {gremium_name_raw}"

        # Enrich analysis metadata if needed
        final_analysis_metadata = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            **analysis_metadata,
        }

        # Enrich generic metadata
        final_metadata = {
            **(metadata or {}),
        }
        if trace_id:
            final_metadata["trace_id"] = trace_id

    audit_meta_payload = dict(audit_meta or {}) if audit_meta is not None else None

    # Create new profile
    create_kwargs = {
        "tenant": tenant,
        "name": name,
        "agreement_type": agreement_type,
        "gremium_identifier": gremium_identifier,
        "gremium_name_raw": gremium_name_raw,
        "version": version,
        "is_current": True,
        "external_id": f"{gremium_identifier}_v{version}",
        "source_document_collection": doc_collection,
        "source_document_id": (UUID(document_id) if document_id else None),
        "structure": structure,
        "analysis_metadata": final_analysis_metadata,
        "metadata": final_metadata,
    }
    if audit_meta_payload is not None:
        create_kwargs["audit_meta"] = audit_meta_payload
    profile = FrameworkProfile.objects.create(**create_kwargs)

    # Create document link for main document
    if document_id:
        FrameworkDocument.objects.create(
            tenant=tenant,
            profile=profile,
            document_collection=doc_collection,
            document_id=UUID(document_id),
            document_type=FrameworkDocument.DocumentType.MAIN,
            position=0,
            analysis={
                "completeness_score": completeness_score,
                "missing_components": missing_components or [],
            },
        )

    logger.info(
        "framework_profile_persisted",
        extra={
            "tenant_schema": tenant_schema,
            "profile_id": str(profile.id),
            "gremium_identifier": gremium_identifier,
            "version": version,
            "is_new_version": existing_profile is not None,
            "agreement_type": agreement_type,
            "trace_id": trace_id,
            # Pre-MVP ID Contract: service identity tracking
            "service_id": (
                audit_meta.get("last_hop_service_id") if audit_meta else None
            ),
            "invocation_id": (audit_meta.get("invocation_id") if audit_meta else None),
        },
    )

    return profile
