"""Test edge cases for workflow_id normalization and storage.

This module tests:
- Case sensitivity preservation
- NFKC Unicode normalization
- Special character rejection
- Cross-layer consistency (contract vs. storage)
"""

import pytest

from documents.contract_utils import normalize_workflow_id
from documents.repository import _workflow_storage_key


class TestWorkflowIdCaseSensitivity:
    """Test that workflow_id preserves case sensitivity across all layers."""

    def test_normalize_workflow_id_preserves_case(self):
        """Contract normalization preserves case."""
        assert normalize_workflow_id("Test") == "Test"
        assert normalize_workflow_id("test") == "test"
        assert normalize_workflow_id("TEST") == "TEST"

    def test_normalize_workflow_id_different_cases_are_distinct(self):
        """Different casing produces different workflow identifiers."""
        workflow_upper = normalize_workflow_id("Project-2024")
        workflow_lower = normalize_workflow_id("project-2024")
        workflow_mixed = normalize_workflow_id("Project-2024")

        assert workflow_upper == "Project-2024"
        assert workflow_lower == "project-2024"
        assert workflow_upper != workflow_lower
        assert workflow_upper == workflow_mixed

    def test_storage_key_preserves_case(self):
        """Storage normalization preserves case."""
        assert _workflow_storage_key("Test") == "Test"
        assert _workflow_storage_key("test") == "test"
        assert _workflow_storage_key("TEST") == "TEST"

    def test_case_consistency_across_layers(self):
        """Case is preserved consistently from contract to storage layer."""
        workflow_input = "Workflow-2024"

        # Contract layer
        contract_normalized = normalize_workflow_id(workflow_input)
        assert contract_normalized == "Workflow-2024"

        # Storage layer (receives output from contract layer)
        storage_key = _workflow_storage_key(contract_normalized)
        assert storage_key == "Workflow-2024"

        # Round trip preserves case
        assert storage_key == workflow_input


class TestWorkflowIdNFKCNormalization:
    """Test NFKC Unicode normalization in workflow_id."""

    def test_ligature_normalization(self):
        """Ligature 'ﬁ' (U+FB01) is normalized to 'fi'."""
        # "proﬁle" with ligature ﬁ
        workflow_ligature = "pro\ufb01le"
        normalized = normalize_workflow_id(workflow_ligature)
        assert normalized == "profile"

    def test_fullwidth_digit_normalization(self):
        """Fullwidth digits (U+FF10-FF19) are normalized to ASCII digits."""
        # "２０２４" (fullwidth)
        workflow_fullwidth = "\uff12\uff10\uff12\uff14"
        normalized = normalize_workflow_id(workflow_fullwidth)
        assert normalized == "2024"

    def test_fullwidth_letter_normalization(self):
        """Fullwidth ASCII letters are normalized to regular ASCII."""
        # "ＴＥＳＴᴬ" (fullwidth + superscript)
        workflow_fullwidth = "\uff34\uff25\uff33\uff34"  # TEST in fullwidth
        normalized = normalize_workflow_id(workflow_fullwidth)
        assert normalized == "TEST"

    def test_combining_characters_normalization(self):
        """Combining characters are normalized via NFKC, but non-ASCII rejected."""
        # "café" with combining acute accent (U+0301)
        workflow_combining = "cafe\u0301"
        # NFKC normalizes to composed form "café", but é is not in [A-Za-z0-9._-]
        with pytest.raises(ValueError, match="workflow_invalid_char"):
            normalize_workflow_id(workflow_combining)

    def test_nfkc_preserves_allowed_characters(self):
        """NFKC normalization doesn't break allowed characters."""
        workflow = "test_2024-v1.0"
        normalized = normalize_workflow_id(workflow)
        assert normalized == "test_2024-v1.0"


class TestWorkflowIdSpecialCharacterRejection:
    """Test that special characters are rejected by normalize_workflow_id."""

    @pytest.mark.parametrize(
        "invalid_workflow_id,reason",
        [
            ("project:2024", "colon not allowed"),
            ("project/v1", "slash not allowed"),
            ("path\\to\\workflow", "backslash not allowed"),
            ("project 2024", "space not allowed"),
            ("user@domain", "at-sign not allowed"),
            ("project#2024", "hash not allowed"),
            ("project$2024", "dollar sign not allowed"),
            ("project%2024", "percent not allowed"),
            ("project&2024", "ampersand not allowed"),
            ("project*2024", "asterisk not allowed"),
            ("project+2024", "plus not allowed"),
            ("project=2024", "equals not allowed"),
            ("project[2024]", "brackets not allowed"),
            ("project{2024}", "braces not allowed"),
            ("project(2024)", "parentheses not allowed"),
            ("project<2024>", "angle brackets not allowed"),
            ("project|2024", "pipe not allowed"),
            ("project;2024", "semicolon not allowed"),
            ("project,2024", "comma not allowed"),
            ("project?2024", "question mark not allowed"),
            ("project!2024", "exclamation not allowed"),
            ("project~2024", "tilde not allowed"),
            ("project`2024", "backtick not allowed"),
            ('project"2024', "double quote not allowed"),
            ("project'2024", "single quote not allowed"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_normalize_workflow_id_rejects_special_characters(
        self, invalid_workflow_id, reason
    ):
        """normalize_workflow_id rejects invalid special characters."""
        with pytest.raises(ValueError, match="workflow_invalid_char"):
            normalize_workflow_id(invalid_workflow_id)

    def test_allowed_special_characters(self):
        """Verify that allowed special characters pass validation."""
        # Only allowed special chars: . _ -
        assert normalize_workflow_id("project.2024") == "project.2024"
        assert normalize_workflow_id("project_2024") == "project_2024"
        assert normalize_workflow_id("project-2024") == "project-2024"
        assert normalize_workflow_id("project.2024-v1_final") == "project.2024-v1_final"


class TestWorkflowIdEdgeCases:
    """Test additional edge cases for workflow_id handling."""

    def test_empty_string_rejected(self):
        """Empty string is rejected."""
        with pytest.raises(ValueError, match="workflow_empty"):
            normalize_workflow_id("")

    def test_whitespace_only_rejected(self):
        """Whitespace-only string is rejected after normalization."""
        with pytest.raises(ValueError, match="workflow_empty"):
            normalize_workflow_id("   ")
        with pytest.raises(ValueError, match="workflow_empty"):
            normalize_workflow_id("\t\n\r")

    def test_invisible_characters_stripped(self):
        """Invisible characters are stripped before validation."""
        # Zero-width space (U+200B) + visible text + zero-width joiner (U+200D)
        workflow_invisible = "\u200b Workflow\u200d-ID \u00a0"
        normalized = normalize_workflow_id(workflow_invisible)
        assert normalized == "Workflow-ID"

    def test_max_length_enforcement(self):
        """Strings exceeding 128 characters are rejected."""
        # Exactly 128 chars (allowed)
        workflow_128 = "a" * 128
        assert normalize_workflow_id(workflow_128) == workflow_128

        # 129 chars (rejected)
        workflow_129 = "a" * 129
        with pytest.raises(ValueError, match="workflow_too_long"):
            normalize_workflow_id(workflow_129)

    def test_storage_key_handles_none(self):
        """Storage key normalization handles None gracefully."""
        assert _workflow_storage_key(None) == ""

    def test_storage_key_handles_empty_string(self):
        """Storage key normalization handles empty string."""
        assert _workflow_storage_key("") == ""
        assert _workflow_storage_key("   ") == ""

    def test_storage_key_strips_whitespace(self):
        """Storage key normalization strips leading/trailing whitespace."""
        assert _workflow_storage_key("  test  ") == "test"
        assert _workflow_storage_key("\ttest\n") == "test"
        assert _workflow_storage_key("  workflow-2024  ") == "workflow-2024"

    def test_unicode_category_edge_cases(self):
        """Test Unicode category edge cases (control chars, format chars)."""
        # Control character (U+0000 NULL) is stripped, leaving "test"
        workflow_with_null = "\x00test"
        normalized = normalize_workflow_id(workflow_with_null)
        assert normalized == "test"  # NULL is stripped by invisible char removal

        # Format character (U+200E LEFT-TO-RIGHT MARK)
        workflow_with_format = "\u200etest\u200e"
        normalized = normalize_workflow_id(workflow_with_format)
        assert normalized == "test"

    def test_alphanumeric_only_accepted(self):
        """Only alphanumeric and [._-] characters are accepted."""
        # Valid alphanumeric + allowed special chars
        assert normalize_workflow_id("abc123") == "abc123"
        assert normalize_workflow_id("ABC123") == "ABC123"
        assert normalize_workflow_id("a1b2c3") == "a1b2c3"
        assert normalize_workflow_id("test.2024-v1_final") == "test.2024-v1_final"

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace is stripped."""
        assert normalize_workflow_id("  test  ") == "test"
        assert normalize_workflow_id("\n\tworkflow-2024\r\n") == "workflow-2024"


class TestWorkflowIdCrossLayerConsistency:
    """Test consistency between contract and storage normalization layers."""

    @pytest.mark.parametrize(
        "input_workflow_id,expected",
        [
            ("test", "test"),
            ("Test", "Test"),
            ("TEST", "TEST"),
            ("project_2024-v1.0", "project_2024-v1.0"),
            ("  workflow  ", "workflow"),
            ("ingestion-2024", "ingestion-2024"),
            ("Case_123.v2", "Case_123.v2"),
        ],
    )
    def test_contract_to_storage_consistency(self, input_workflow_id, expected):
        """Workflow ID remains consistent from contract to storage layer."""
        # Step 1: Contract normalization
        contract_normalized = normalize_workflow_id(input_workflow_id.strip())

        # Step 2: Storage normalization
        storage_key = _workflow_storage_key(contract_normalized)

        # Both should produce the same result
        assert contract_normalized == expected
        assert storage_key == expected

    def test_storage_key_is_idempotent(self):
        """Storage key normalization is idempotent."""
        workflow = "test-2024"
        key1 = _workflow_storage_key(workflow)
        key2 = _workflow_storage_key(key1)
        key3 = _workflow_storage_key(key2)

        assert key1 == key2 == key3 == workflow

    def test_contract_validation_before_storage(self):
        """Contract validation should happen before storage normalization."""
        # This workflow_id would fail contract validation
        invalid = "invalid:workflow"

        # Contract layer rejects it
        with pytest.raises(ValueError, match="workflow_invalid_char"):
            normalize_workflow_id(invalid)

        # Storage layer would accept it (no validation, just normalization)
        # This shows why contract validation MUST happen first
        storage_result = _workflow_storage_key(invalid)
        assert storage_result == "invalid:workflow"  # Accepts anything
