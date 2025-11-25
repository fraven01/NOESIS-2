"""Unit tests for framework analysis utility functions."""

from __future__ import annotations


from ai_core.graphs.framework_analysis_graph import (
    normalize_gremium_identifier,
    extract_toc_from_chunks,
)


class TestGremiumNormalization:
    """Tests for gremium identifier normalization."""

    def test_normalize_kbr_basic(self) -> None:
        """Test basic KBR normalization."""
        result = normalize_gremium_identifier("KBR", "Konzernbetriebsrat")
        assert result == "KBR"

    def test_normalize_kbr_full_name(self) -> None:
        """Test KBR with full German name."""
        result = normalize_gremium_identifier(
            "Konzernbetriebsrat",
            "Konzernbetriebsrat der Telefónica Deutschland Holding AG",
        )
        assert result == "KONZERNBETRIEBSRAT"

    def test_normalize_gbr_with_location(self) -> None:
        """Test GBR with location."""
        result = normalize_gremium_identifier(
            "GBR München", "Gesamtbetriebsrat München"
        )
        assert result == "GBR_MUENCHEN"

    def test_normalize_br_with_location(self) -> None:
        """Test BR with location."""
        result = normalize_gremium_identifier(
            "BR Berlin", "Betriebsrat Berlin Werk Nord"
        )
        assert result == "BR_BERLIN"

    def test_normalize_with_umlauts(self) -> None:
        """Test normalization with German umlauts."""
        result = normalize_gremium_identifier("BR Düsseldorf", "Betriebsrat Düsseldorf")
        assert result == "BR_DUESSELDORF"

    def test_normalize_multiple_umlauts(self) -> None:
        """Test normalization with multiple umlauts."""
        result = normalize_gremium_identifier(
            "BR Köln-Mülheim", "Betriebsrat Köln-Mülheim"
        )
        # Should handle ö→OE and remove hyphens
        assert result == "BR_KOELN_MUELHEIM"

    def test_normalize_with_special_chars(self) -> None:
        """Test normalization with special characters."""
        result = normalize_gremium_identifier(
            "BR-Werk/Nord (Standort 1)", "Betriebsrat Werk Nord Standort 1"
        )
        # Should replace all special chars with underscores
        assert result == "BR_WERK_NORD_STANDORT_1"

    def test_normalize_consecutive_underscores(self) -> None:
        """Test that consecutive underscores are merged."""
        result = normalize_gremium_identifier(
            "BR   Multiple   Spaces", "Betriebsrat Multiple Spaces"
        )
        # Multiple spaces should become single underscore
        assert result == "BR_MULTIPLE_SPACES"

    def test_normalize_leading_trailing_underscores(self) -> None:
        """Test that leading/trailing underscores are removed."""
        result = normalize_gremium_identifier(" BR Leading ", "Betriebsrat Leading")
        assert result == "BR_LEADING"
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_normalize_lowercase_to_uppercase(self) -> None:
        """Test that lowercase is converted to uppercase."""
        result = normalize_gremium_identifier("kbr", "Konzernbetriebsrat")
        assert result == "KBR"

    def test_normalize_mixed_case(self) -> None:
        """Test mixed case normalization."""
        result = normalize_gremium_identifier(
            "GbR München", "Gesamtbetriebsrat München"
        )
        assert result == "GBR_MUENCHEN"


class TestTocExtraction:
    """Tests for table of contents extraction from chunks."""

    def test_extract_toc_from_empty_chunks(self) -> None:
        """Test ToC extraction from empty chunk list."""
        result = extract_toc_from_chunks([])
        assert result == []

    def test_extract_toc_from_chunks_without_parents(self) -> None:
        """Test ToC extraction from chunks without parent metadata."""
        chunks = [
            {"text": "Content", "meta": {}},
            {"text": "More content", "meta": {"other": "data"}},
        ]
        result = extract_toc_from_chunks(chunks)
        assert result == []

    def test_extract_toc_with_single_heading(self) -> None:
        """Test ToC extraction with a single heading."""
        chunks = [
            {
                "text": "Content",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h1",
                            "type": "heading",
                            "title": "§ 1 Geltungsbereich",
                            "level": 1,
                            "order": 1,
                        }
                    ]
                },
            }
        ]
        result = extract_toc_from_chunks(chunks)
        assert len(result) == 1
        assert result[0]["id"] == "doc#h1"
        assert result[0]["title"] == "§ 1 Geltungsbereich"
        assert result[0]["level"] == 1
        assert result[0]["order"] == 1

    def test_extract_toc_with_hierarchy(self) -> None:
        """Test ToC extraction with hierarchical structure."""
        chunks = [
            {
                "text": "Content 1",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h1",
                            "type": "heading",
                            "title": "§ 1 Präambel",
                            "level": 1,
                            "order": 1,
                        }
                    ]
                },
            },
            {
                "text": "Content 2",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h2",
                            "type": "heading",
                            "title": "§ 2 Geltungsbereich",
                            "level": 1,
                            "order": 2,
                        },
                        {
                            "id": "doc#h2.1",
                            "type": "heading",
                            "title": "2.1 Persönlicher Geltungsbereich",
                            "level": 2,
                            "order": 3,
                        },
                    ]
                },
            },
        ]
        result = extract_toc_from_chunks(chunks)
        assert len(result) == 3
        # Should be sorted by level and order
        assert result[0]["title"] == "§ 1 Präambel"
        assert result[1]["title"] == "§ 2 Geltungsbereich"
        assert result[2]["title"] == "2.1 Persönlicher Geltungsbereich"

    def test_extract_toc_deduplication(self) -> None:
        """Test that duplicate parents are deduplicated."""
        chunks = [
            {
                "text": "Content 1",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h1",
                            "type": "heading",
                            "title": "§ 1 Geltungsbereich",
                            "level": 1,
                            "order": 1,
                        }
                    ]
                },
            },
            {
                "text": "Content 2",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h1",
                            "type": "heading",
                            "title": "§ 1 Geltungsbereich",
                            "level": 1,
                            "order": 1,
                        }
                    ]
                },
            },
        ]
        result = extract_toc_from_chunks(chunks)
        assert len(result) == 1  # Should be deduplicated

    def test_extract_toc_filters_non_structural(self) -> None:
        """Test that non-structural parent types are filtered out."""
        chunks = [
            {
                "text": "Content",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h1",
                            "type": "heading",
                            "title": "§ 1 Geltungsbereich",
                            "level": 1,
                            "order": 1,
                        },
                        {
                            "id": "doc#p1",
                            "type": "paragraph",
                            "title": "Some paragraph",
                            "level": 2,
                            "order": 2,
                        },
                        {
                            "id": "doc#s1",
                            "type": "section",
                            "title": "Section 1",
                            "level": 1,
                            "order": 3,
                        },
                    ]
                },
            }
        ]
        result = extract_toc_from_chunks(chunks)
        # Should only include heading and section, not paragraph
        assert len(result) == 2
        types = {entry["type"] for entry in result}
        assert "heading" in types
        assert "section" in types
        assert "paragraph" not in types

    def test_extract_toc_sorting(self) -> None:
        """Test that ToC entries are sorted correctly."""
        chunks = [
            {
                "text": "Content",
                "meta": {
                    "parents": [
                        {
                            "id": "doc#h3",
                            "type": "heading",
                            "title": "§ 3 Third",
                            "level": 1,
                            "order": 30,
                        },
                        {
                            "id": "doc#h1",
                            "type": "heading",
                            "title": "§ 1 First",
                            "level": 1,
                            "order": 10,
                        },
                        {
                            "id": "doc#h2",
                            "type": "heading",
                            "title": "§ 2 Second",
                            "level": 1,
                            "order": 20,
                        },
                    ]
                },
            }
        ]
        result = extract_toc_from_chunks(chunks)
        # Should be sorted by order
        assert result[0]["title"] == "§ 1 First"
        assert result[1]["title"] == "§ 2 Second"
        assert result[2]["title"] == "§ 3 Third"
