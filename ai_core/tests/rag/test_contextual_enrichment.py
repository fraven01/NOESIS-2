from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.llm.client import LlmClientError
from ai_core.rag.contextual_enrichment import (
    ContextualEnrichmentConfig,
    generate_contextual_prefixes,
)
from ai_core.tool_contracts.base import tool_context_from_scope
from documents.pipeline import DocumentProcessingContext, DocumentProcessingMetadata


def _build_context():
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-ctx",
        invocation_id=str(uuid4()),
        run_id="run-ctx",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    business = BusinessContext(case_id="case-ctx")
    return tool_context_from_scope(scope, business=business)


def test_generate_contextual_prefixes_respects_limits(monkeypatch) -> None:
    def fake_call(model, prompt, metadata):
        return {"text": "This chunk summarizes the payment terms section."}

    monkeypatch.setattr("ai_core.rag.contextual_enrichment.llm_client.call", fake_call)
    config = ContextualEnrichmentConfig(
        enabled=True,
        model_label="fast",
        max_document_chars=100,
        max_chunk_chars=50,
        max_chunks=1,
        max_prefix_chars=200,
        max_prefix_words=10,
    )
    entries = [{"text": "First chunk text"}, {"text": "Second chunk text"}]
    prefixes = generate_contextual_prefixes(
        "Document text",
        entries,
        _build_context(),
        config,
    )

    assert prefixes[0] == "This chunk summarizes the payment terms section."
    assert prefixes[1] is None


def test_generate_contextual_prefixes_handles_llm_error(monkeypatch) -> None:
    def fake_call(model, prompt, metadata):
        raise LlmClientError("boom")

    monkeypatch.setattr("ai_core.rag.contextual_enrichment.llm_client.call", fake_call)
    config = ContextualEnrichmentConfig(
        enabled=True,
        model_label="fast",
        max_document_chars=100,
        max_chunk_chars=50,
        max_chunks=10,
        max_prefix_chars=200,
        max_prefix_words=10,
    )
    prefixes = generate_contextual_prefixes(
        "Document text",
        [{"text": "Chunk"}],
        _build_context(),
        config,
    )

    assert prefixes == [None]


def test_generate_contextual_prefixes_accepts_processing_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call(model, prompt, metadata):
        captured.update(metadata)
        return {"text": "Document scope summary."}

    monkeypatch.setattr("ai_core.rag.contextual_enrichment.llm_client.call", fake_call)
    metadata = DocumentProcessingMetadata(
        tenant_id="tenant-ctx",
        collection_id=None,
        document_collection_id=None,
        case_id="case-ctx",
        workflow_id="workflow-ctx",
        document_id=uuid4(),
        version="v1",
        source="test",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        trace_id="trace-proc",
    )
    context = DocumentProcessingContext(metadata=metadata)
    config = ContextualEnrichmentConfig(
        enabled=True,
        model_label="fast",
        max_document_chars=100,
        max_chunk_chars=50,
        max_chunks=1,
        max_prefix_chars=200,
        max_prefix_words=10,
    )
    prefixes = generate_contextual_prefixes(
        "Document text",
        [{"text": "Chunk"}],
        context,
        config,
    )

    assert prefixes == ["Document scope summary."]
    assert captured.get("tenant_id") == "tenant-ctx"
    assert captured.get("case_id") == "case-ctx"
    assert captured.get("trace_id") == "trace-proc"
