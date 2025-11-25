"""Tests for framework analysis contracts."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from ai_core.tools.framework_contracts import (
    TypeDetectionOutput,
    TypeEvidence,
    ComponentLocation,
    ComponentValidation,
    FrameworkAnalysisInput,
    FrameworkAnalysisOutput,
    FrameworkStructure,
    AssembledComponentLocation,
    FrameworkAnalysisMetadata,
)


class TestTypeDetectionOutput:
    """Tests for TypeDetectionOutput contract."""

    def test_valid_type_detection(self) -> None:
        """Test valid type detection output."""
        output = TypeDetectionOutput(
            agreement_type="kbv",
            type_confidence=0.95,
            gremium_name_raw="Konzernbetriebsrat der Telefónica Deutschland",
            gremium_identifier_suggestion="KBR",
            evidence=[
                TypeEvidence(
                    text="Konzernbetriebsvereinbarung",
                    location="Präambel",
                    reasoning="Explicit mention",
                )
            ],
            scope_indicators={"raeumlich": "alle Betriebe", "sachlich": "IT-Systeme"},
        )
        assert output.agreement_type == "kbv"
        assert output.type_confidence == 0.95

    def test_invalid_agreement_type(self) -> None:
        """Test that invalid agreement types are rejected."""
        with pytest.raises(ValidationError):
            TypeDetectionOutput(
                agreement_type="invalid_type",  # Not in allowed literals
                type_confidence=0.95,
                gremium_name_raw="Test",
                gremium_identifier_suggestion="TEST",
                evidence=[],
                scope_indicators={},
            )

    def test_confidence_bounds(self) -> None:
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            TypeDetectionOutput(
                agreement_type="kbv",
                type_confidence=1.5,  # > 1.0
                gremium_name_raw="Test",
                gremium_identifier_suggestion="TEST",
                evidence=[],
                scope_indicators={},
            )

        with pytest.raises(ValidationError):
            TypeDetectionOutput(
                agreement_type="kbv",
                type_confidence=-0.1,  # < 0.0
                gremium_name_raw="Test",
                gremium_identifier_suggestion="TEST",
                evidence=[],
                scope_indicators={},
            )

    def test_evidence_max_length(self) -> None:
        """Test that evidence fields respect max length."""
        with pytest.raises(ValidationError):
            TypeEvidence(text="x" * 201, location="Test", reasoning="Test")  # Max 200


class TestComponentLocation:
    """Tests for ComponentLocation contract."""

    def test_component_location_main(self) -> None:
        """Test component location in main document."""
        location = ComponentLocation(
            location="main",
            outline_path="2",
            heading="§ 2 Systembeschreibung",
            chunk_ids=["chunk1", "chunk2"],
            page_numbers=[2, 3],
            confidence=0.92,
        )
        assert location.location == "main"
        assert location.outline_path == "2"

    def test_component_location_annex(self) -> None:
        """Test component location in annex."""
        location = ComponentLocation(
            location="annex",
            outline_path="Anlage 1",
            candidate_annex="Anlage 1",
            chunk_ids=["chunk5"],
            page_numbers=[15],
            confidence=0.88,
        )
        assert location.location == "annex"
        assert location.candidate_annex == "Anlage 1"

    def test_component_location_not_found(self) -> None:
        """Test component not found."""
        location = ComponentLocation(location="not_found", confidence=0.0)
        assert location.location == "not_found"
        assert location.confidence == 0.0
        assert location.chunk_ids == []

    def test_invalid_location_type(self) -> None:
        """Test that invalid location types are rejected."""
        with pytest.raises(ValidationError):
            ComponentLocation(location="invalid_location", confidence=0.5)


class TestFrameworkAnalysisInput:
    """Tests for FrameworkAnalysisInput contract."""

    def test_valid_input(self) -> None:
        """Test valid analysis input."""
        input_params = FrameworkAnalysisInput(
            document_collection_id=uuid4(),
            document_id=uuid4(),
            force_reanalysis=False,
            confidence_threshold=0.70,
        )
        assert isinstance(input_params.document_collection_id, UUID)
        assert input_params.confidence_threshold == 0.70

    def test_input_defaults(self) -> None:
        """Test input with default values."""
        input_params = FrameworkAnalysisInput(document_collection_id=uuid4())
        assert input_params.document_id is None
        assert input_params.force_reanalysis is False
        assert input_params.confidence_threshold == 0.70

    def test_confidence_threshold_bounds(self) -> None:
        """Test confidence threshold validation."""
        with pytest.raises(ValidationError):
            FrameworkAnalysisInput(
                document_collection_id=uuid4(), confidence_threshold=1.5  # > 1.0
            )


class TestFrameworkStructure:
    """Tests for FrameworkStructure contract."""

    def test_valid_structure(self) -> None:
        """Test valid framework structure."""
        structure = FrameworkStructure(
            systembeschreibung=AssembledComponentLocation(
                location="main",
                outline_path="2",
                heading="§ 2 Systembeschreibung",
                chunk_ids=["chunk1"],
                page_numbers=[2],
                confidence=0.92,
                validated=True,
                validation_notes="Plausible",
            ),
            funktionsbeschreibung=AssembledComponentLocation(
                location="annex",
                outline_path="Anlage 1",
                chunk_ids=[],
                page_numbers=[],
                confidence=0.88,
                validated=True,
            ),
            auswertungen=AssembledComponentLocation(
                location="annex_group",
                outline_path="Anlage 3",
                annex_root="Anlage 3",
                subannexes=["3.1", "3.2"],
                chunk_ids=[],
                page_numbers=[],
                confidence=0.85,
                validated=True,
            ),
            zugriffsrechte=AssembledComponentLocation(
                location="not_found",
                chunk_ids=[],
                page_numbers=[],
                confidence=0.0,
                validated=False,
            ),
        )
        assert structure.systembeschreibung.location == "main"
        assert structure.auswertungen.subannexes == ["3.1", "3.2"]
        assert not structure.zugriffsrechte.validated


class TestFrameworkAnalysisOutput:
    """Tests for FrameworkAnalysisOutput contract."""

    def test_valid_output(self) -> None:
        """Test valid analysis output."""
        profile_id = uuid4()
        output = FrameworkAnalysisOutput(
            profile_id=profile_id,
            version=1,
            gremium_identifier="KBR",
            structure=FrameworkStructure(
                systembeschreibung=AssembledComponentLocation(
                    location="main",
                    chunk_ids=[],
                    page_numbers=[],
                    confidence=0.9,
                    validated=True,
                ),
                funktionsbeschreibung=AssembledComponentLocation(
                    location="main",
                    chunk_ids=[],
                    page_numbers=[],
                    confidence=0.9,
                    validated=True,
                ),
                auswertungen=AssembledComponentLocation(
                    location="main",
                    chunk_ids=[],
                    page_numbers=[],
                    confidence=0.9,
                    validated=True,
                ),
                zugriffsrechte=AssembledComponentLocation(
                    location="not_found",
                    chunk_ids=[],
                    page_numbers=[],
                    confidence=0.0,
                    validated=False,
                ),
            ),
            completeness_score=0.75,
            missing_components=["zugriffsrechte"],
            hitl_required=False,
            hitl_reasons=[],
            analysis_metadata=FrameworkAnalysisMetadata(
                detected_type="kbv",
                type_confidence=0.95,
                gremium_name_raw="Konzernbetriebsrat",
                gremium_identifier="KBR",
                completeness_score=0.75,
                missing_components=["zugriffsrechte"],
                analysis_timestamp="2025-01-15T10:00:00Z",
            ),
        )
        assert output.profile_id == profile_id
        assert output.completeness_score == 0.75
        assert "zugriffsrechte" in output.missing_components

    def test_output_immutability(self) -> None:
        """Test that output contracts are frozen."""
        metadata = FrameworkAnalysisMetadata(
            detected_type="kbv",
            type_confidence=0.95,
            gremium_name_raw="Test",
            gremium_identifier="KBR",
            completeness_score=0.75,
            missing_components=[],
            analysis_timestamp="2025-01-15T10:00:00Z",
        )

        # Frozen models should raise error on field assignment
        with pytest.raises((ValidationError, AttributeError)):
            metadata.detected_type = "gbv"  # type: ignore[misc]


class TestComponentValidation:
    """Tests for ComponentValidation contract."""

    def test_plausible_validation(self) -> None:
        """Test plausible component validation."""
        validation = ComponentValidation(
            component="systembeschreibung",
            plausible=True,
            confidence=0.91,
            reason="Contains technical specifications",
        )
        assert validation.plausible
        assert validation.confidence == 0.91
        assert validation.why_not is None

    def test_implausible_validation(self) -> None:
        """Test implausible component validation."""
        validation = ComponentValidation(
            component="zugriffsrechte",
            plausible=False,
            confidence=0.35,
            reason="Describes legal framework",
            why_not="Expected roles/permissions matrix missing",
        )
        assert not validation.plausible
        assert validation.why_not is not None

    def test_validation_with_warnings(self) -> None:
        """Test validation with warnings."""
        validation = ComponentValidation(
            component="auswertungen",
            plausible=True,
            confidence=0.75,
            reason="Contains report listings",
            warnings=["Very short section", "No detailed examples"],
        )
        assert len(validation.warnings) == 2
