"""Contracts for framework agreement analysis."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Framework Type Detection
# ============================================================================


class TypeEvidence(BaseModel):
    """Evidence for framework type classification."""

    text: str = Field(..., max_length=200)
    location: str = Field(..., max_length=255)
    reasoning: str = Field(..., max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True)


class AlternativeType(BaseModel):
    """Alternative framework type interpretation."""

    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScopeIndicators(BaseModel):
    """Scope indicators for framework agreement."""

    raeumlich: str | None = None  # Spatial scope
    sachlich: str | None = None  # Subject scope

    model_config = ConfigDict(extra="forbid", frozen=True)


class TypeDetectionOutput(BaseModel):
    """Output from framework type and gremium detection."""

    agreement_type: Literal["kbv", "gbv", "bv", "dv", "other"]
    type_confidence: float = Field(..., ge=0.0, le=1.0)
    gremium_name_raw: str = Field(..., max_length=255)
    gremium_identifier_suggestion: str = Field(..., max_length=128)
    evidence: list[TypeEvidence]
    scope_indicators: ScopeIndicators
    alternative_types: list[AlternativeType] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


# ============================================================================
# Component Location
# ============================================================================


class LocationEvidence(BaseModel):
    """Evidence for component location."""

    structural: str | None = None
    semantic: str | None = None
    parent_title: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentLocation(BaseModel):
    """Location of a framework component."""

    location: Literal["main", "annex", "annex_group", "not_found"]
    outline_path: str | None = None
    heading: str | None = None
    candidate_path: str | None = None
    candidate_annex: str | None = None
    annex_root: str | None = None
    subannex_pattern: str | None = None
    subannexes: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: LocationEvidence | None = None
    searched_locations: list[str] = Field(default_factory=list)
    recommendation: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentLocationsOutput(BaseModel):
    """Output from component location analysis."""

    systembeschreibung: ComponentLocation
    funktionsbeschreibung: ComponentLocation
    auswertungen: ComponentLocation
    zugriffsrechte: ComponentLocation

    model_config = ConfigDict(extra="forbid", frozen=True)


# ============================================================================
# Component Validation
# ============================================================================


class ContextAnalysis(BaseModel):
    """Analysis of surrounding context."""

    before: str | None = None
    after: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentValidation(BaseModel):
    """Validation result for a component."""

    component: Literal[
        "systembeschreibung",
        "funktionsbeschreibung",
        "auswertungen",
        "zugriffsrechte",
    ]
    plausible: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., max_length=1000)
    why_not: str | None = Field(None, max_length=1000)
    context_analysis: ContextAnalysis | None = None
    warnings: list[str] = Field(default_factory=list)
    alternative_interpretation: str | None = Field(None, max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentValidationsOutput(BaseModel):
    """Output from component validation."""

    validations: list[ComponentValidation]

    model_config = ConfigDict(extra="forbid", frozen=True)


# ============================================================================
# Framework Analysis Inputs/Outputs
# ============================================================================


class FrameworkAnalysisInput(BaseModel):
    """Input for framework analysis."""

    document_collection_id: UUID
    document_id: UUID | None = None
    force_reanalysis: bool = False
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class AssembledComponentLocation(BaseModel):
    """Assembled and validated component location."""

    location: Literal["main", "annex", "annex_group", "not_found"]
    outline_path: str | None = None
    heading: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    validated: bool
    validation_notes: str | None = None
    annex_root: str | None = None
    subannexes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameworkStructure(BaseModel):
    """Complete framework structure with all components."""

    systembeschreibung: AssembledComponentLocation
    funktionsbeschreibung: AssembledComponentLocation
    auswertungen: AssembledComponentLocation
    zugriffsrechte: AssembledComponentLocation

    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameworkAnalysisMetadata(BaseModel):
    """Metadata from framework analysis."""

    detected_type: str
    type_confidence: float = Field(..., ge=0.0, le=1.0)
    gremium_name_raw: str
    gremium_identifier: str
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    missing_components: list[str]
    analysis_timestamp: str
    model_version: str = "framework_analysis_v1"

    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameworkAnalysisOutput(BaseModel):
    """Output from framework analysis."""

    profile_id: UUID
    version: int
    gremium_identifier: str
    structure: FrameworkStructure
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    missing_components: list[str]
    hitl_required: bool
    hitl_reasons: list[str] = Field(default_factory=list)
    idempotent: bool = True
    analysis_metadata: FrameworkAnalysisMetadata

    model_config = ConfigDict(extra="forbid", frozen=True)


# ============================================================================
# Error Codes
# ============================================================================


class FrameworkAnalysisErrorCode:
    """Machine-readable framework analysis error codes."""

    DOCUMENT_NOT_FOUND = "FRAMEWORK_DOCUMENT_NOT_FOUND"
    COLLECTION_NOT_FOUND = "FRAMEWORK_COLLECTION_NOT_FOUND"
    PROFILE_EXISTS = "FRAMEWORK_PROFILE_EXISTS"
    ANALYSIS_FAILED = "FRAMEWORK_ANALYSIS_FAILED"
    LLM_ERROR = "FRAMEWORK_LLM_ERROR"
    VALIDATION_FAILED = "FRAMEWORK_VALIDATION_FAILED"


def map_framework_error_to_status(code: str) -> int:
    """Return HTTP status mapped to a framework analysis error code."""
    if code in {
        FrameworkAnalysisErrorCode.DOCUMENT_NOT_FOUND,
        FrameworkAnalysisErrorCode.COLLECTION_NOT_FOUND,
    }:
        return 404
    if code == FrameworkAnalysisErrorCode.PROFILE_EXISTS:
        return 409  # Conflict
    return 500  # Internal error for analysis/LLM failures
