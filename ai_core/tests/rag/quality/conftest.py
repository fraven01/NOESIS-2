"""Test fixtures for RAG quality metrics tests."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_llm_judge():
    """Mock LLM for testing LLM-as-Judge quality scoring."""

    def _mock_completion(model: str, messages: List[dict], **kwargs) -> Any:
        """Return fake LLM response with quality scores."""
        # Extract chunk text from prompt
        prompt = messages[0]["content"]

        # Fake quality scores based on chunk characteristics
        chunk_length = len(prompt)

        # Longer chunks generally score higher (but with variance)
        base_score = min(90, chunk_length // 10 + 50)

        coherence = base_score + (hash(prompt) % 10 - 5)
        completeness = base_score + (hash(prompt[:20]) % 10 - 5)
        reference_resolution = base_score - 5 + (hash(prompt[20:40]) % 10 - 5)
        redundancy = max(0, min(20, 100 - chunk_length // 20))

        # Clamp to 0-100
        coherence = max(0, min(100, coherence))
        completeness = max(0, min(100, completeness))
        reference_resolution = max(0, min(100, reference_resolution))
        redundancy = max(0, min(100, redundancy))

        # Return mock response with structured JSON
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[
            0
        ].message.content = f"""{{
            "coherence": {coherence},
            "completeness": {completeness},
            "reference_resolution": {reference_resolution},
            "redundancy": {redundancy},
            "reasoning": "Test chunk evaluation based on length and hash"
        }}"""

        return response

    return _mock_completion


@pytest.fixture
def mock_pseudo_query_generator():
    """Mock LLM for testing pseudo query generation."""

    def _mock_completion(model: str, messages: List[dict], **kwargs) -> Any:
        """Return fake LLM response with pseudo queries."""
        # Extract chunk text from prompt
        prompt = messages[0]["content"]

        # Generate fake queries based on chunk hash
        chunk_hash = hash(prompt)
        queries = [
            f"What is the main topic discussed in this chunk? (hash: {chunk_hash % 1000})",
            f"How does this relate to the overall document? (hash: {(chunk_hash >> 10) % 1000})",
            f"What are the key points mentioned? (hash: {(chunk_hash >> 20) % 1000})",
        ]

        # Return mock response with structured JSON
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = f'{{"queries": {queries}}}'

        return response

    return _mock_completion


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing quality metrics."""
    return [
        {
            "chunk_id": "test-chunk-1",
            "text": "This is a high-quality chunk with clear semantic meaning. It is self-contained and provides complete information about the topic.",
            "parent_ref": "section:introduction",
            "metadata": {
                "section_path": ["Introduction"],
                "kind": "paragraph",
            },
        },
        {
            "chunk_id": "test-chunk-2",
            "text": "Short chunk.",
            "parent_ref": "section:background",
            "metadata": {
                "section_path": ["Background"],
                "kind": "paragraph",
            },
        },
        {
            "chunk_id": "test-chunk-3",
            "text": "This chunk has many references to previous sections. It mentions the above concept multiple times. The aforementioned idea is crucial.",
            "parent_ref": "section:discussion",
            "metadata": {
                "section_path": ["Discussion"],
                "kind": "paragraph",
            },
        },
    ]


@pytest.fixture
def sample_quality_scores():
    """Sample quality scores for testing."""
    from ai_core.rag.quality.llm_judge import ChunkQualityScore

    return [
        ChunkQualityScore(
            chunk_id="uuid-1",
            coherence=90,
            completeness=85,
            reference_resolution=80,
            redundancy=95,
            overall=87.5,
        ),
        ChunkQualityScore(
            chunk_id="uuid-2",
            coherence=60,
            completeness=50,
            reference_resolution=40,
            redundancy=70,
            overall=55.0,
        ),
        ChunkQualityScore(
            chunk_id="uuid-3",
            coherence=75,
            completeness=70,
            reference_resolution=50,
            redundancy=80,
            overall=68.75,
        ),
    ]


@pytest.fixture
def sample_pseudo_queries():
    """Sample pseudo queries with weak labels."""
    return [
        {
            "chunk_id": "uuid-1",
            "queries": [
                "What is semantic chunking?",
                "How does late chunking work?",
                "Why is contextual embedding important?",
            ],
            "document_id": "doc-1",
            "tenant_id": "tenant-1",
        },
        {
            "chunk_id": "uuid-2",
            "queries": [
                "What are the benefits of agentic chunking?",
                "How do LLMs detect boundaries?",
            ],
            "document_id": "doc-1",
            "tenant_id": "tenant-1",
        },
    ]


@pytest.fixture
def sample_golden_set():
    """Sample golden set with validated query-answer pairs."""
    return [
        {
            "query": "What is semantic chunking?",
            "true_chunk_id": "uuid-1",
            "document_id": "doc-1",
            "validated_by": "user-1",
            "validated_at": "2025-12-29T12:00:00Z",
        },
        {
            "query": "How does late chunking work?",
            "true_chunk_id": "uuid-1",
            "document_id": "doc-1",
            "validated_by": "user-1",
            "validated_at": "2025-12-29T12:01:00Z",
        },
    ]
