"""Contracts for framework agreement analysis."""

from __future__ import annotations

from typing import Any, Literal
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

    def is_strong(self, *, min_reasoning_chars: int = 30) -> bool:
        """Return True when reasoning meets a minimal explanatory bar."""
        return len(self.reasoning.strip()) >= min_reasoning_chars

    def merge_with(self, other: TypeEvidence) -> TypeEvidence:
        """Merge reasoning when text/location match."""
        if self.text != other.text or self.location != other.location:
            raise ValueError("Cannot merge evidence with different text/location")
        if self.reasoning == other.reasoning:
            return self
        merged_reasoning = self._merge_reasoning(self.reasoning, other.reasoning)
        return self.model_copy(update={"reasoning": merged_reasoning})

    @staticmethod
    def _merge_reasoning(first: str, second: str) -> str:
        if not first:
            return second
        if not second:
            return first
        if second in first:
            return first
        if first in second:
            return second
        return f"{first}; {second}"


class AlternativeType(BaseModel):
    """Alternative framework type interpretation."""

    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def is_confident(self, *, min_confidence: float = 0.7) -> bool:
        """Return True when confidence meets a minimal threshold."""
        return self.confidence >= min_confidence

    def normalized_type(self) -> str:
        """Return a normalized type token for comparisons."""
        return self.type.strip().lower()


class ScopeIndicators(BaseModel):
    """Scope indicators for framework agreement."""

    raeumlich: str | None = None  # Spatial scope
    sachlich: str | None = None  # Subject scope

    model_config = ConfigDict(extra="forbid", frozen=True)

    def has_any(self) -> bool:
        """Return True when any scope indicator is present."""
        return bool((self.raeumlich or "").strip() or (self.sachlich or "").strip())

    def merge_with(self, other: ScopeIndicators) -> ScopeIndicators:
        """Prefer existing values and fill missing fields from other."""
        return self.model_copy(
            update={
                "raeumlich": self.raeumlich or other.raeumlich,
                "sachlich": self.sachlich or other.sachlich,
            }
        )


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

    @classmethod
    def from_partial(cls, data: dict[str, Any] | None) -> ComponentLocation:
        """Build a location from partial data with safe defaults."""
        payload = dict(data or {})
        location = payload.get("location")
        if location is None or location == "":
            payload["location"] = "not_found"
        if payload.get("confidence") is None:
            payload["confidence"] = 0.0
        return cls.model_validate(payload)

    def is_found(self) -> bool:
        """Return True when the component is located."""
        return self.location != "not_found"

    def is_low_confidence(self, threshold: float) -> bool:
        """Return True when confidence is below the threshold."""
        return self.confidence < threshold

    def validation_summary(
        self, *, high_confidence_threshold: float = 0.8
    ) -> dict[str, Any]:
        """Return a lightweight validation summary for the location."""
        if not self.is_found():
            return {"plausible": False, "confidence": 0.0, "reason": "Not found"}
        if self.confidence >= high_confidence_threshold:
            return {
                "plausible": True,
                "confidence": self.confidence,
                "reason": "High confidence",
            }
        return {
            "plausible": True,
            "confidence": self.confidence,
            "reason": "Moderate confidence",
        }

    def to_assembled(
        self,
        *,
        validation: dict[str, Any] | None = None,
        default_validation_reason: str = "Plausible",
    ) -> tuple[AssembledComponentLocation, bool]:
        """Build an assembled location and flag validation failures."""
        if not self.is_found():
            return (
                AssembledComponentLocation(
                    location="not_found",
                    outline_path=None,
                    heading=None,
                    chunk_ids=[],
                    page_numbers=[],
                    confidence=0.0,
                    validated=False,
                    validation_notes="Not found",
                    annex_root=None,
                    subannexes=[],
                ),
                False,
            )

        validation_payload = validation or {}
        plausible = validation_payload.get("plausible", True)
        reason = validation_payload.get(
            "reason",
            default_validation_reason if plausible else "",
        )
        assembled = AssembledComponentLocation(
            location=self.location,
            outline_path=self.outline_path,
            heading=self.heading,
            chunk_ids=self.chunk_ids,
            page_numbers=self.page_numbers,
            confidence=self.confidence,
            validated=plausible,
            validation_notes=reason,
            annex_root=self.annex_root,
            subannexes=self.subannexes,
        )
        return assembled, not plausible


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
    """Input for framework analysis.

    BREAKING CHANGE (Option A - Strict Separation):
    Business domain IDs (document_collection_id, document_id) have been REMOVED.
    These are now read from ToolContext.business.

    Golden Rule: Tool-Inputs contain only functional parameters.
    Context contains Scope, Business, and Runtime Permissions.
    """

    force_reanalysis: bool = False
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class FrameworkAnalysisDraftMetadata(BaseModel):
    """Metadata captured before persistence."""

    detected_type: str
    type_confidence: float = Field(..., ge=0.0, le=1.0)
    gremium_name_raw: str
    gremium_identifier: str
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    missing_components: list[str]
    model_version: str = "framework_analysis_v1"

    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameworkAnalysisError(BaseModel):
    """Structured error payload for framework analysis."""

    node: str
    message: str
    error_type: str

    model_config = ConfigDict(extra="forbid", frozen=True)


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


class FrameworkAnalysisDraft(BaseModel):
    """Pre-persistence framework analysis output."""

    gremium_identifier: str
    gremium_name_raw: str
    agreement_type: str
    document_collection_id: str
    document_id: str | None
    structure: FrameworkStructure
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    missing_components: list[str]
    hitl_required: bool
    hitl_reasons: list[str] = Field(default_factory=list)
    analysis_metadata: FrameworkAnalysisDraftMetadata
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    force_reanalysis: bool = False
    partial_results: FrameworkStructure | None = None
    errors: list[FrameworkAnalysisError] = Field(default_factory=list)

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
