"""Test fixtures for RAG chunking tests."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stub_embedding_client():
    """Stub embedding client for testing Late Chunking without LiteLLM."""

    class StubEmbeddingClient:
        """Stub that returns fake embeddings based on text hash."""

        def __init__(self, dimension: int = 1536):
            self.dimension = dimension

        def embed(self, texts: List[str], model: str = None) -> Any:
            """Return fake embeddings (magnitude based on text hash)."""
            _ = model  # Ignore model parameter in stub
            embeddings = []
            for text in texts:
                # Create deterministic fake embedding
                magnitude = hash(text) % 100 / 100.0  # 0.0-1.0
                embedding = [magnitude] + [0.0] * (self.dimension - 1)
                embeddings.append(embedding)

            # Return mock response with vectors attribute
            response = MagicMock()
            response.vectors = embeddings
            return response

    return StubEmbeddingClient()


@pytest.fixture
def mock_llm_boundary_detection():
    """Mock LLM for testing Agentic Chunking boundary detection."""

    def _mock_completion(model: str, messages: List[dict], **kwargs) -> Any:
        """Return fake LLM response with boundary detection."""
        # Extract document text from prompt
        prompt = messages[0]["content"]

        # Fake boundary detection: split at every 500 characters
        doc_length = len(prompt)
        boundaries = []
        current = 0

        while current < doc_length:
            end = min(current + 500, doc_length)
            boundaries.append(
                {
                    "start": current,
                    "end": end,
                    "reason": "Semantic boundary detected (mocked)",
                }
            )
            current = end

        # Return mock response with structured JSON
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = f'{{"boundaries": {boundaries}}}'
        response.usage = MagicMock()
        response.usage.total_tokens = 100

        return response

    return _mock_completion


@pytest.fixture
def mock_llm_quality_evaluation():
    """Mock LLM for testing chunk quality evaluation."""

    def _mock_completion(model: str, messages: List[dict], **kwargs) -> Any:
        """Return fake LLM response with quality scores."""
        # Extract chunk text from prompt
        prompt = messages[0]["content"]

        # Fake quality scores based on chunk length (longer = better)
        chunk_length = len(prompt)
        coherence = min(100, chunk_length // 10 + 50)
        completeness = min(100, chunk_length // 10 + 60)
        reference_resolution = min(100, chunk_length // 10 + 55)
        redundancy = max(0, 100 - chunk_length // 20)

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
            "reasoning": "Mocked quality evaluation"
        }}"""

        return response

    return _mock_completion


@pytest.fixture
def sample_parsed_result():
    """Sample ParsedResult for testing chunkers."""
    from documents.pipeline import ParsedResult, ParsedTextBlock

    blocks = [
        ParsedTextBlock(
            kind="heading",
            text="Introduction",
            section_path=("Introduction",),
            page_index=None,
        ),
        ParsedTextBlock(
            kind="paragraph",
            text="This is a sample paragraph for testing. " * 10,
            section_path=("Introduction",),
            page_index=None,
        ),
        ParsedTextBlock(
            kind="heading",
            text="Background",
            section_path=("Background",),
            page_index=None,
        ),
        ParsedTextBlock(
            kind="paragraph",
            text="Background information here. " * 15,
            section_path=("Background",),
            page_index=None,
        ),
    ]

    return ParsedResult(
        text_blocks=blocks,
        assets=[],
        statistics={"block.count": len(blocks)},
    )


@pytest.fixture
def sample_processing_context():
    """Sample DocumentProcessingContext for testing."""
    from documents.pipeline import DocumentProcessingContext, DocumentProcessingMetadata
    from uuid import uuid4
    from datetime import datetime, timezone

    metadata = DocumentProcessingMetadata(
        tenant_id="test-tenant",
        document_id=uuid4(),
        workflow_id="test-workflow",
        created_at=datetime.now(timezone.utc),
        trace_id="test-trace-001",
        span_id="test-span-001",
    )

    return DocumentProcessingContext(
        metadata=metadata,
        trace_id="test-trace-001",
        span_id="test-span-001",
    )


@pytest.fixture
def sample_pipeline_config():
    """Sample DocumentPipelineConfig for testing."""
    from documents.pipeline import DocumentPipelineConfig

    return DocumentPipelineConfig()
