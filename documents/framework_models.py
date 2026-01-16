"""Framework agreement models for IT co-determination."""

from __future__ import annotations

import uuid

from django.db import models


class FrameworkAgreementType(models.TextChoices):
    """Types of framework agreements for co-determination."""

    KBV = "kbv", "Konzernbetriebsvereinbarung"
    GBV = "gbv", "Gesamtbetriebsvereinbarung"
    BV = "bv", "Betriebsvereinbarung"
    DV = "dv", "Dienstvereinbarung"
    OTHER = "other", "Other"


class FrameworkProfile(models.Model):
    """
    Framework agreement profile with multi-version support.
    Each gremium can have multiple versions (old ones stay active for existing cases).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="framework_profiles",
    )

    # Identification
    name = models.CharField(max_length=255)
    agreement_type = models.CharField(
        max_length=32,
        choices=FrameworkAgreementType.choices,
        default=FrameworkAgreementType.OTHER,
    )

    # Gremium identification (AI-extracted, normalized)
    gremium_identifier = models.CharField(max_length=128)
    """
    AI-extracted and normalized:
    - "KBR" | "Konzernbetriebsrat" | "Konzern-BR" → "KBR"
    - "GBR München" | "Gesamtbetriebsrat München" → "GBR_MUENCHEN"
    - "BR Berlin" | "Betriebsrat Berlin" → "BR_BERLIN"
    """

    gremium_name_raw = models.CharField(max_length=255)
    """AI-extracted original name from document."""

    # Versioning (old frameworks stay active for existing cases)
    version = models.IntegerField(default=1)
    valid_from = models.DateField(null=True, blank=True)
    valid_until = models.DateField(null=True, blank=True)  # NULL = unlimited validity
    is_current = models.BooleanField(default=True)
    """True = most recent framework for this gremium."""

    # External reference
    external_id = models.CharField(max_length=128)

    # Source document reference
    source_document_collection = models.ForeignKey(
        "documents.DocumentCollection",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="framework_profiles_source",
    )
    source_document_id = models.UUIDField(null=True, blank=True)

    # Structure definition (JSON)
    structure = models.JSONField(blank=True, default=dict)
    """
    Structure of the framework agreement:
    {
      "systembeschreibung": {
        "location": "main|annex|annex_group|not_found",
        "outline_path": "2",
        "chunk_ids": ["..."],
        "page_numbers": [2, 3],
        "heading": "2. Systembeschreibung",
        "confidence": 0.92,
        "validated": true
      },
      "funktionsbeschreibung": {...},
      "auswertungen": {...},
      "zugriffsrechte": {...}
    }
    """

    # Analysis metadata
    analysis_metadata = models.JSONField(blank=True, default=dict)
    """
    Metadata from framework analysis:
    {
      "detected_type": "kbv",
      "type_confidence": 0.95,
      "gremium_name_raw": "Konzernbetriebsrat der Telefónica Deutschland",
      "gremium_identifier": "KBR",
      "completeness_score": 0.75,
      "missing_components": ["zugriffsrechte"],
      "analysis_timestamp": "2025-01-15T10:30:00Z",
      "model_version": "framework_analysis_v1"
    }
    """

    # Metadata
    audit_meta = models.JSONField(blank=True, default=dict)
    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant", "gremium_identifier", "version"),
                name="framework_profile_unique_version",
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant", "agreement_type"),
                name="fp_tenant_type_idx",
            ),
            models.Index(
                fields=("tenant", "gremium_identifier", "is_current"),
                name="fp_tenant_gremium_idx",
            ),
            models.Index(
                fields=("source_document_id",),
                name="fp_source_doc_idx",
            ),
        ]
        ordering = ["-version", "-created_at"]

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.name} v{self.version} ({self.gremium_identifier})"


class FrameworkDocument(models.Model):
    """Links documents (main + annexes) to a framework profile."""

    class DocumentType(models.TextChoices):
        MAIN = "main", "Main Agreement"
        ANNEX = "annex", "Annex/Attachment"
        PROTOCOL = "protocol", "Signing Protocol"
        AMENDMENT = "amendment", "Amendment"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.PROTECT,
        related_name="framework_documents",
    )
    profile = models.ForeignKey(
        FrameworkProfile,
        on_delete=models.CASCADE,
        related_name="documents",
    )

    # Document reference
    document_collection = models.ForeignKey(
        "documents.DocumentCollection",
        on_delete=models.CASCADE,
        related_name="framework_documents",
    )
    document_id = models.UUIDField()

    # Position in framework
    document_type = models.CharField(
        max_length=32,
        choices=DocumentType.choices,
        default=DocumentType.MAIN,
    )
    position = models.IntegerField(default=0)  # Sort order
    annex_number = models.CharField(max_length=32, blank=True, default="")

    # Analysis results (populated by planner)
    analysis = models.JSONField(blank=True, default=dict)
    """
    Analysis results for this document:
    {
      "detected_sections": [
        {"id": "scope", "found": true, "location": "page 2, § 1"}
      ],
      "completeness_score": 0.95,
      "missing_sections": [],
      "cross_references": [...]
    }
    """

    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(
                fields=("tenant", "profile"),
                name="fd_tenant_profile_idx",
            ),
            models.Index(
                fields=("document_id",),
                name="framework_doc_document_id_idx",
            ),
        ]
        ordering = ["position"]

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.document_type} {self.annex_number} (Profile: {self.profile_id})"
