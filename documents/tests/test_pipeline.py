from __future__ import annotations

import base64
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import import_module
from typing import Optional
from uuid import uuid4

import pytest

from documents import (
    DocumentChunkArtifact,
    DocumentParseArtifact,
    DocumentPipelineConfig,
    DocumentProcessingPhase,
    DocumentProcessingOrchestrator,
    DocumentProcessingContext,
    DocumentProcessingMetadata,
    DocumentProcessingOutcome,
    ParsedResult,
    ProcessingState,
    persist_parsed_document,
    require_document_components,
    require_document_contracts,
)
from documents.captioning import DeterministicCaptioner
from documents import metrics
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    FileBlob,
    NormalizedDocument,
)
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage
from documents.parsers import (
    build_parsed_asset,
    build_parsed_result,
    build_parsed_text_block,
)


_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFElEQVR4nGOsfq7LgA0wYRUdtBIAO/8Bn565mEEAAAAASUVORK5CYII="
_PNG_BYTES = base64.b64decode(_PNG_BASE64)


def _sample_document() -> NormalizedDocument:
    tenant_id = "tenant-1"
    workflow_id = "workflow-1"
    document_id = uuid4()
    collection_id = uuid4()
    document_collection_id = uuid4()
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        collection_id=collection_id,
        document_collection_id=document_collection_id,
        version="v1",
    )
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_collection_id=document_collection_id,
    )

    # Use FileBlob instead of InlineBlob
    blob = FileBlob(
        type="file",
        uri="memory://sample-doc",
        sha256="b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        size=11,
    )
    created_at = datetime.now(timezone.utc)
    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=blob.sha256,
        created_at=created_at,
        source="upload",
    )


def _prepare_repository_document():
    storage = InMemoryStorage()

    # Prime storage with the content referenced by _sample_document
    # "hello world" -> b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    storage._store["memory://sample-doc"] = b"hello world"

    repository = InMemoryDocumentsRepository(storage=storage)
    stored = repository.upsert(_sample_document())
    context = DocumentProcessingContext.from_document(stored)
    return repository, storage, stored, context


class RecordingParser:
    def __init__(self, *, statistics: Optional[dict[str, object]] = None) -> None:
        self.calls = 0
        self.statistics = statistics or {"ocr.triggered_pages": [1], "parser.words": 2}

    def can_handle(self, document: object) -> bool:
        return True

    def parse(self, document: object, config: object) -> ParsedResult:
        self.calls += 1
        payload = b"asset"
        return build_parsed_result(
            text_blocks=[build_parsed_text_block(text="Block text", kind="paragraph")],
            assets=[build_parsed_asset(media_type="image/png", content=payload)],
            statistics=dict(self.statistics),
        )


class RecordingChunker:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def chunk(
        self,
        document: object,
        parsed: ParsedResult,
        *,
        context: DocumentProcessingContext,
        config: DocumentPipelineConfig,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("chunk_failure")
        chunks = [
            {"text": block.text, "ord": index}
            for index, block in enumerate(parsed.text_blocks)
        ]
        return chunks, {"chunk.count": len(chunks)}


def test_pipeline_config_defaults():
    config = DocumentPipelineConfig()
    assert config.pdf_safe_mode is True
    assert config.caption_min_confidence_default == pytest.approx(0.4)
    assert config.caption_min_confidence_by_collection == {}
    assert config.enable_ocr is False
    assert config.enable_notes_in_pptx is True
    assert config.emit_empty_slides is True
    assert config.enable_asset_captions is True
    assert config.ocr_fallback_confidence == pytest.approx(0.5)


def test_pipeline_config_collection_override():
    collection_uuid = uuid4()
    config = DocumentPipelineConfig(
        caption_min_confidence_default=0.4,
        caption_min_confidence_by_collection={
            "collection-1": 0.7,
            collection_uuid: 0.9,
            "  MixedCase  ": 0.6,
        },
    )
    assert config.caption_min_confidence("collection-1") == pytest.approx(0.7)
    assert config.caption_min_confidence(collection_uuid) == pytest.approx(0.9)
    assert config.caption_min_confidence(str(collection_uuid)) == pytest.approx(0.9)
    assert config.caption_min_confidence("mixedcase") == pytest.approx(0.6)
    assert config.caption_min_confidence("unknown") == pytest.approx(0.4)


def test_pipeline_config_normalization_idempotent():
    config = DocumentPipelineConfig(
        caption_min_confidence_by_collection={"  COLLECTION  ": 0.55}
    )
    rehydrated = DocumentPipelineConfig(
        caption_min_confidence_by_collection=config.caption_min_confidence_by_collection
    )
    assert (
        config.caption_min_confidence_by_collection
        == rehydrated.caption_min_confidence_by_collection
    )


def test_pipeline_config_flag_combinations():
    config = DocumentPipelineConfig(
        pdf_safe_mode=False,
        enable_ocr=True,
        enable_notes_in_pptx=False,
        emit_empty_slides=False,
        enable_asset_captions=False,
        ocr_fallback_confidence=0.25,
    )
    assert config.pdf_safe_mode is False
    assert config.enable_ocr is True
    assert config.enable_notes_in_pptx is False
    assert config.emit_empty_slides is False
    assert config.enable_asset_captions is False
    assert config.ocr_fallback_confidence == pytest.approx(0.25)


def test_processing_phase_coerce_aliases():
    assert (
        DocumentProcessingPhase.coerce("preview")
        is DocumentProcessingPhase.PARSE_AND_PERSIST
    )
    assert (
        DocumentProcessingPhase.coerce("Review")
        is DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION
    )
    assert (
        DocumentProcessingPhase.coerce("parse-only")
        is DocumentProcessingPhase.PARSE_ONLY
    )
    assert (
        DocumentProcessingPhase.coerce(DocumentProcessingPhase.FULL)
        is DocumentProcessingPhase.FULL
    )
    assert DocumentProcessingPhase.coerce(None) is DocumentProcessingPhase.FULL


@pytest.mark.parametrize(
    "value",
    [0.0, 0.5, 1.0],
)
def test_pipeline_config_confidence_bounds(value):
    config = DocumentPipelineConfig(
        caption_min_confidence_default=value,
        caption_min_confidence_by_collection={"collection": value},
        ocr_fallback_confidence=value,
    )
    assert config.caption_min_confidence_default == pytest.approx(value)
    assert config.caption_min_confidence("collection") == pytest.approx(value)
    assert config.ocr_fallback_confidence == pytest.approx(value)


@pytest.mark.parametrize(
    "bad_value",
    [-0.01, 1.01],
)
def test_pipeline_config_confidence_out_of_range(bad_value):
    with pytest.raises(ValueError) as exc:
        DocumentPipelineConfig(
            caption_min_confidence_default=bad_value,
        )
    assert str(exc.value) == "caption_min_confidence_default_range"

    with pytest.raises(ValueError) as exc:
        DocumentPipelineConfig(
            caption_min_confidence_by_collection={"collection": bad_value},
        )
    assert str(exc.value) == "caption_min_confidence_range"

    with pytest.raises(ValueError) as exc:
        DocumentPipelineConfig(
            ocr_fallback_confidence=bad_value,
        )
    assert str(exc.value) == "ocr_fallback_confidence_range"


@pytest.mark.parametrize("key", ["", "   "])
def test_pipeline_config_rejects_empty_collection_key_in_mapping(key):
    with pytest.raises(ValueError) as exc:
        DocumentPipelineConfig(
            caption_min_confidence_by_collection={key: 0.5},
        )
    assert str(exc.value) == "caption_min_confidence_collection_empty"


@pytest.mark.parametrize("key", ["", "   "])
def test_pipeline_config_lookup_with_empty_collection_key_falls_back(key):
    config = DocumentPipelineConfig(
        caption_min_confidence_default=0.3,
        caption_min_confidence_by_collection={"collection": 0.6},
    )
    assert config.caption_min_confidence(key) == pytest.approx(0.3)


def test_processing_context_roundtrip_preserves_metadata():
    document = _sample_document()
    metadata = DocumentProcessingMetadata.from_document(document, case_id="case-1")
    context = DocumentProcessingContext(metadata=metadata)

    for state in (
        ProcessingState.PARSED_TEXT,
        ProcessingState.ASSETS_EXTRACTED,
        ProcessingState.CAPTIONED,
        ProcessingState.CHUNKED,
    ):
        context = context.transition(state)
        assert context.metadata is metadata
        assert context.metadata.tenant_id == document.ref.tenant_id
        assert context.metadata.collection_id == document.ref.collection_id
        assert (
            context.metadata.document_collection_id
            == document.meta.document_collection_id
        )
        assert context.metadata.workflow_id == document.ref.workflow_id
        assert context.metadata.document_id == document.ref.document_id
        assert context.metadata.version == document.ref.version
        assert context.metadata.source == document.source
        assert context.metadata.created_at == document.created_at


def test_processing_context_with_case_override_preserves_values():
    document = _sample_document()
    context = DocumentProcessingContext.from_document(
        document,
        case_id="  CASE-123  ",
        initial_state=ProcessingState.PARSED_TEXT,
        trace_id=" trace-xyz ",
        span_id=" span-uvw ",
    )

    assert context.state is ProcessingState.PARSED_TEXT
    assert context.metadata.case_id == "CASE-123"
    assert context.trace_id == "trace-xyz"
    assert context.span_id == "span-uvw"

    for next_state in (
        ProcessingState.ASSETS_EXTRACTED,
        ProcessingState.CAPTIONED,
        ProcessingState.CHUNKED,
    ):
        context = context.transition(next_state)
        assert context.metadata.case_id == "CASE-123"
        assert context.metadata.workflow_id == document.ref.workflow_id
        assert context.metadata.collection_id == document.ref.collection_id
        assert (
            context.metadata.document_collection_id
            == document.meta.document_collection_id
        )
        assert context.trace_id == "trace-xyz"
        assert context.span_id == "span-uvw"


def test_processing_context_repr_contains_state_and_metadata():
    document = _sample_document()
    context = DocumentProcessingContext.from_document(document)
    representation = repr(context)
    assert "DocumentProcessingContext" in representation
    assert f"state='{context.state.value}'" in representation
    assert str(document.ref.document_id) in representation


def test_orchestrator_binds_trace_and_span_in_log_context(monkeypatch):
    metrics.reset_metrics()
    repository, storage, document, _ = _prepare_repository_document()
    parser = RecordingParser()
    chunker = RecordingChunker()
    orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=chunker,
    )

    captured: list[dict[str, object]] = []

    @contextmanager
    def _capture_log_context(**kwargs):
        captured.append(kwargs)
        yield

    monkeypatch.setattr("documents.pipeline.log_context", _capture_log_context)

    context = DocumentProcessingContext.from_document(
        document,
        trace_id="trace-log",
        span_id="span-log",
    )

    orchestrator.process(document, context=context)

    assert captured, "expected log_context to be invoked"
    bound = captured[0]
    assert bound["trace_id"] == "trace-log"
    assert bound["span_id"] == "span-log"
    assert bound["tenant"] == context.metadata.tenant_id


def test_require_document_contracts(monkeypatch):
    contracts = require_document_contracts()
    assert contracts.normalized_document is NormalizedDocument
    assert contracts.document_ref is DocumentRef
    assert contracts.asset.__name__ == "Asset"
    assert contracts.asset_ref.__name__ == "AssetRef"

    from documents import contracts as contracts_module

    monkeypatch.delattr(contracts_module, "AssetRef")
    with pytest.raises(RuntimeError) as exc:
        require_document_contracts()
    assert str(exc.value) == "contract_missing_asset_ref"


def test_require_document_contracts_missing_module(monkeypatch):
    real_import = import_module

    def _raise_for_contracts(name):
        if name == "documents.contracts":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr("documents.pipeline.import_module", _raise_for_contracts)

    with pytest.raises(RuntimeError) as exc:
        require_document_contracts()

    assert str(exc.value) == "contract_missing_documents_contracts_module"


def test_require_document_components(monkeypatch):
    components = require_document_components()
    from documents import captioning, repository, storage

    assert components.repository is repository.DocumentsRepository
    assert components.storage is storage.Storage
    assert components.captioner is captioning.MultimodalCaptioner

    monkeypatch.setattr(repository, "DocumentsRepository", None)
    with pytest.raises(RuntimeError) as exc:
        require_document_components()
    assert str(exc.value) == "contract_missing_documents_repository"


def test_require_document_components_missing_module(monkeypatch):
    real_import = import_module

    def _raise_for_repository(name):
        if name in {"documents.repository", "documents.repositories"}:
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr("documents.pipeline.import_module", _raise_for_repository)

    with pytest.raises(RuntimeError) as exc:
        require_document_components()

    assert str(exc.value) == "contract_missing_documents_repository_module"


def test_metadata_created_at_requires_timezone():
    with pytest.raises(ValueError) as exc:
        DocumentProcessingMetadata(
            tenant_id="tenant",
            collection_id=None,
            case_id=None,
            workflow_id="workflow",
            document_id=uuid4(),
            version=None,
            source=None,
            created_at=datetime.now(),
        )

    error = exc.value.errors()[0]["ctx"]["error"].args[0]
    assert error == "metadata_created_at_naive"


def test_metadata_uses_workflow_from_meta_when_missing_on_ref():
    document_id = uuid4()
    created_at = datetime.now(timezone.utc)
    document = type(
        "DocStub",
        (),
        {
            "ref": type(
                "RefStub",
                (),
                {
                    "tenant_id": " tenant\u200b-1 ",
                    "workflow_id": None,
                    "document_id": document_id,
                    "collection_id": None,
                    "version": None,
                },
            )(),
            "meta": type(
                "MetaStub",
                (),
                {"workflow_id": "workflow-1"},
            )(),
            "created_at": created_at,
            "source": "upload",
        },
    )()

    metadata = DocumentProcessingMetadata.from_document(document, case_id=" case ")
    assert metadata.workflow_id == "workflow-1"
    assert metadata.tenant_id == "tenant-1"
    assert metadata.case_id == "case"
    assert metadata.created_at == created_at


def test_metadata_coalesces_trace_and_span_sources():
    document_id = uuid4()
    created_at = datetime.now(timezone.utc)

    class _GraphState:
        trace_id = " trace-123 "
        span_id = " span-456 "

    document = type(
        "DocStub",
        (),
        {
            "ref": type(
                "RefStub",
                (),
                {
                    "tenant_id": "tenant-1",
                    "workflow_id": "workflow-1",
                    "document_id": document_id,
                    "collection_id": None,
                    "version": None,
                },
            )(),
            "meta": type(
                "MetaStub",
                (),
                {
                    "workflow_id": "workflow-1",
                    "graph_state": _GraphState(),
                },
            )(),
            "created_at": created_at,
            "source": "upload",
        },
    )()

    metadata = DocumentProcessingMetadata.from_document(document)

    assert metadata.trace_id == "trace-123"
    assert metadata.span_id == "span-456"


def test_persist_parsed_document_stores_assets_and_stats():
    repository, storage, document, context = _prepare_repository_document()
    payload = _PNG_BYTES
    parsed = build_parsed_result(
        text_blocks=[build_parsed_text_block(text="Hello world", kind="paragraph")],
        assets=[build_parsed_asset(media_type="image/png", content=payload)],
        statistics={"parser.words": 2},
    )

    artefact = persist_parsed_document(
        context,
        document,
        parsed,
        repository=repository,
        storage=storage,
    )

    assert isinstance(artefact, DocumentParseArtifact)
    assert artefact.parsed_context.state is ProcessingState.PARSED_TEXT
    assert artefact.asset_context.state is ProcessingState.ASSETS_EXTRACTED
    assert artefact.text_blocks[0]["text"] == "Hello world"
    assert artefact.statistics["parser.words"] == 2
    assert artefact.statistics["parse.assets.total"] == 1
    assert artefact.statistics["parse.assets.media_type.image_png"] == 1

    asset_ref = artefact.asset_refs[0]
    stored_asset = repository.get_asset(
        context.metadata.tenant_id,
        asset_ref.asset_id,
        workflow_id=context.metadata.workflow_id,
    )
    assert stored_asset is not None
    assert stored_asset.checksum == hashlib.sha256(payload).hexdigest()
    assert storage.get(stored_asset.blob.uri) == payload
    assert stored_asset.caption_source == "none"
    assert stored_asset.caption_method == "none"
    assert stored_asset.caption_confidence is None
    assert stored_asset.parent_ref is None
    assert stored_asset.perceptual_hash is not None

    updated_document = repository.get(
        context.metadata.tenant_id,
        context.metadata.document_id,
        context.metadata.version,
        workflow_id=context.metadata.workflow_id,
    )
    assert updated_document is not None
    assert updated_document.meta.parse_stats["parser.words"] == 2
    assert (
        updated_document.meta.parse_stats["parse.blocks.total"]
        == artefact.statistics["parse.blocks.total"]
    )


def test_persist_parsed_document_is_idempotent():
    repository, storage, document, context = _prepare_repository_document()
    payload = _PNG_BYTES
    parsed = build_parsed_result(
        text_blocks=[build_parsed_text_block(text="Block", kind="paragraph")],
        assets=[build_parsed_asset(media_type="image/png", content=payload)],
        statistics={},
    )

    first = persist_parsed_document(
        context,
        document,
        parsed,
        repository=repository,
        storage=storage,
    )
    asset_ref = first.asset_refs[0]
    stored_first = repository.get_asset(
        context.metadata.tenant_id,
        asset_ref.asset_id,
        workflow_id=context.metadata.workflow_id,
    )
    assert stored_first is not None
    uri_before = stored_first.blob.uri

    second = persist_parsed_document(
        context,
        document,
        parsed,
        repository=repository,
        storage=storage,
    )
    stored_second = repository.get_asset(
        context.metadata.tenant_id,
        second.asset_refs[0].asset_id,
        workflow_id=context.metadata.workflow_id,
    )
    assert stored_second is not None
    assert stored_second.blob.uri == uri_before
    assert first.asset_refs[0].asset_id == stored_first.ref.asset_id
    assert second.asset_refs[0].asset_id == stored_first.ref.asset_id
    refs, _ = repository.list_assets_by_document(
        context.metadata.tenant_id,
        context.metadata.document_id,
        workflow_id=context.metadata.workflow_id,
    )
    assert len(refs) == 1


def test_persist_parsed_document_supports_external_uri_assets():
    repository, storage, document, context = _prepare_repository_document()
    file_uri = "https://example.com/image.png"
    parsed = build_parsed_result(
        text_blocks=[],
        assets=[build_parsed_asset(media_type="image/png", file_uri=file_uri)],
        statistics={},
    )

    artefact = persist_parsed_document(
        context,
        document,
        parsed,
        repository=repository,
        storage=storage,
    )

    stored_asset = repository.get_asset(
        context.metadata.tenant_id,
        artefact.asset_refs[0].asset_id,
        workflow_id=context.metadata.workflow_id,
    )
    assert stored_asset is not None
    assert stored_asset.blob.type == "external"
    assert stored_asset.origin_uri == file_uri
    assert stored_asset.blob.kind == "https"
    assert stored_asset.blob.uri == file_uri
    assert stored_asset.checksum == hashlib.sha256(file_uri.encode("utf-8")).hexdigest()
    assert stored_asset.caption_source == "none"
    assert stored_asset.caption_method == "none"
    assert stored_asset.caption_confidence is None
    assert stored_asset.parent_ref is None
    assert stored_asset.perceptual_hash is None


def test_orchestrator_executes_full_pipeline():
    metrics.reset_metrics()
    repository, storage, document, _ = _prepare_repository_document()
    parser = RecordingParser()
    chunker = RecordingChunker()
    orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=chunker,
    )

    outcome = orchestrator.process(document)

    assert isinstance(outcome, DocumentProcessingOutcome)
    assert outcome.context.state is ProcessingState.CHUNKED
    assert isinstance(outcome.parse_artifact, DocumentParseArtifact)
    assert isinstance(outcome.chunk_artifact, DocumentChunkArtifact)
    assert outcome.chunk_artifact.statistics["chunk.count"] == 1

    stored = repository.get(
        outcome.context.metadata.tenant_id,
        outcome.context.metadata.document_id,
        outcome.context.metadata.version,
        workflow_id=outcome.context.metadata.workflow_id,
    )
    assert stored is not None
    stats = stored.meta.parse_stats or {}
    assert stats["parse.state"] == ProcessingState.PARSED_TEXT.value
    assert stats["assets.state"] == ProcessingState.ASSETS_EXTRACTED.value
    assert stats["caption.state"] == ProcessingState.CAPTIONED.value
    assert stats["chunk.state"] == ProcessingState.CHUNKED.value
    assert pytest.approx(stats["caption.hit_rate"]) == 1.0
    assert parser.calls == 1
    assert chunker.calls == 1

    workflow_label = outcome.context.metadata.workflow_id
    assert (
        metrics.counter_value(metrics.PIPELINE_BLOCKS_TOTAL, workflow_id=workflow_label)
        == 1.0
    )
    assert (
        metrics.counter_value(metrics.PIPELINE_ASSETS_TOTAL, workflow_id=workflow_label)
        == 1.0
    )
    assert (
        metrics.counter_value(
            metrics.PIPELINE_OCR_TRIGGER_TOTAL, workflow_id=workflow_label
        )
        == 1.0
    )
    assert (
        metrics.counter_value(
            metrics.PIPELINE_CAPTION_HITS_TOTAL, workflow_id=workflow_label
        )
        == 1.0
    )
    assert (
        metrics.counter_value(
            metrics.PIPELINE_CAPTION_ATTEMPTS_TOTAL, workflow_id=workflow_label
        )
        == 1.0
    )
    assert (
        metrics.histogram_count(
            metrics.PIPELINE_CAPTION_HIT_RATIO, workflow_id=workflow_label
        )
        == 1.0
    )


def test_orchestrator_preview_phase_stops_after_persist():
    metrics.reset_metrics()
    repository, storage, document, _ = _prepare_repository_document()
    parser = RecordingParser()
    chunker = RecordingChunker()
    orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=chunker,
    )

    outcome = orchestrator.process(document, run_until="preview")

    assert isinstance(outcome.parse_artifact, DocumentParseArtifact)
    assert outcome.chunk_artifact is None
    assert outcome.context.state is ProcessingState.ASSETS_EXTRACTED
    stats = outcome.document.meta.parse_stats or {}
    assert stats["parse.state"] == ProcessingState.PARSED_TEXT.value
    assert stats["assets.state"] == ProcessingState.ASSETS_EXTRACTED.value
    assert "caption.state" not in stats
    assert chunker.calls == 0


def test_orchestrator_review_phase_stops_after_caption():
    metrics.reset_metrics()
    repository, storage, document, _ = _prepare_repository_document()
    parser = RecordingParser()
    chunker = RecordingChunker()
    orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=chunker,
    )

    outcome = orchestrator.process(
        document,
        run_until=DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION,
    )

    assert isinstance(outcome.parse_artifact, DocumentParseArtifact)
    assert outcome.chunk_artifact is None
    assert outcome.context.state is ProcessingState.CAPTIONED
    stats = outcome.document.meta.parse_stats or {}
    assert stats["caption.state"] == ProcessingState.CAPTIONED.value
    assert "chunk.state" not in stats
    assert chunker.calls == 0


def test_orchestrator_is_idempotent():
    metrics.reset_metrics()
    repository, storage, document, _ = _prepare_repository_document()
    parser = RecordingParser()
    chunker = RecordingChunker()
    orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=chunker,
    )

    first = orchestrator.process(document)
    assert first.context.state is ProcessingState.CHUNKED
    assert parser.calls == 1
    assert chunker.calls == 1

    stored = repository.get(
        first.context.metadata.tenant_id,
        first.context.metadata.document_id,
        first.context.metadata.version,
        workflow_id=first.context.metadata.workflow_id,
    )
    assert stored is not None

    second = orchestrator.process(stored)
    assert isinstance(second, DocumentProcessingOutcome)
    # Without delta decider/caching, pipeline re-processes safe idempotent path
    assert second.parse_artifact is not None
    assert second.chunk_artifact is not None
    assert second.context.state is ProcessingState.CHUNKED
    assert parser.calls == 2
    assert chunker.calls == 2


def test_orchestrator_handles_chunk_failures():
    metrics.reset_metrics()
    repository, storage, document, _ = _prepare_repository_document()
    parser = RecordingParser()
    failing_chunker = RecordingChunker(fail=True)
    orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=failing_chunker,
    )

    with pytest.raises(RuntimeError):
        orchestrator.process(document)

    assert parser.calls == 1
    assert failing_chunker.calls == 1

    stored_after_failure = repository.get(
        document.ref.tenant_id,
        document.ref.document_id,
        document.ref.version,
        workflow_id=document.ref.workflow_id,
    )
    assert stored_after_failure is not None
    stats = stored_after_failure.meta.parse_stats or {}
    assert stats["assets.state"] == ProcessingState.ASSETS_EXTRACTED.value
    assert "chunk.state" not in stats

    recovery_chunker = RecordingChunker()
    retry_orchestrator = DocumentProcessingOrchestrator(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=DeterministicCaptioner(),
        chunker=recovery_chunker,
    )

    outcome = retry_orchestrator.process(stored_after_failure)
    assert outcome.context.state is ProcessingState.CHUNKED
    assert parser.calls == 2
    assert recovery_chunker.calls == 1
    final_stats = outcome.document.meta.parse_stats or {}
    assert final_stats["chunk.state"] == ProcessingState.CHUNKED.value
