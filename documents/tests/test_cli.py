import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import get_args
from uuid import uuid4

import pytest

from documents.captioning import DeterministicCaptioner
from documents.cli import CLIContext, main
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.parsers import (
    ParsedResult,
    ParserDispatcher,
    ParserRegistry,
    build_parsed_asset,
    build_parsed_result,
    build_parsed_text_block,
)
from documents.pipeline import DocumentPipelineConfig, ProcessingState
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage


class StubParser:
    def __init__(self) -> None:
        self.calls = 0

    def can_handle(self, document: object) -> bool:
        return True

    def parse(self, document: object, config: object) -> ParsedResult:
        self.calls += 1
        image_payload = b"image-bytes"
        text_block = build_parsed_text_block(text="Sample paragraph", kind="paragraph")
        asset = build_parsed_asset(media_type="image/png", content=image_payload)
        statistics = {"parser.words": 2}
        return build_parsed_result(
            text_blocks=[text_block],
            assets=[asset],
            statistics=statistics,
        )


def _sample_document() -> NormalizedDocument:
    tenant_id = "tenant-cli"
    workflow_id = "workflow-cli"
    document_id = uuid4()
    collection_id = uuid4()
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        collection_id=collection_id,
        version="v1",
    )
    meta = DocumentMeta(tenant_id=tenant_id, workflow_id=workflow_id)
    payload = b"# Heading\n\nContent"
    encoded = base64.b64encode(payload).decode("ascii")
    discriminator_field = "type" if "type" in InlineBlob.model_fields else "kind"
    literal_values = get_args(InlineBlob.model_fields[discriminator_field].annotation)
    discriminator_value = literal_values[0]
    blob = InlineBlob.model_validate(
        {
            discriminator_field: discriminator_value,
            "media_type": "text/markdown",
            "base64": encoded,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
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


def _build_context(parser: StubParser) -> CLIContext:
    from ai_core.rag.chunking import HybridChunker, ChunkerConfig, ChunkerMode

    storage = InMemoryStorage()
    repository = InMemoryDocumentsRepository(storage=storage)
    registry = ParserRegistry([parser])
    dispatcher = ParserDispatcher(registry)
    captioner = DeterministicCaptioner()
    chunker_config = ChunkerConfig(
        mode=ChunkerMode.LATE,
        enable_quality_metrics=False,  # Disable for CLI test performance
    )
    chunker = HybridChunker(chunker_config)
    config = DocumentPipelineConfig(caption_min_confidence_default=0.0)
    return CLIContext(
        repository=repository,
        storage=storage,
        parser_registry=registry,
        parser=dispatcher,
        captioner=captioner,
        chunker=chunker,
        config=config,
    )


def _repository_document(context: CLIContext) -> NormalizedDocument:
    stored = context.repository.upsert(_sample_document())
    assert stored is not None
    return stored


@pytest.mark.parametrize("json_output", [True, False])
def test_cli_parse_caption_chunk_smoke(capsys, json_output: bool) -> None:
    parser = StubParser()
    context = _build_context(parser)
    document = _repository_document(context)

    # Always use --json because we need to parse the output as JSON
    # The json_output parameter controls the explicit flag, but CLI always outputs JSON
    common_args = ["--json"] if json_output else []
    parse_args = common_args + [
        "parse",
        "--tenant",
        document.ref.tenant_id,
        "--doc-id",
        str(document.ref.document_id),
        "--workflow-id",
        document.ref.workflow_id,
    ]
    exit_code = main(parse_args, context=context)
    capture = capsys.readouterr()

    # Debug: if no output, check stderr
    if not capture.out.strip():
        if capture.err.strip():
            pytest.fail(f"CLI wrote to stderr instead of stdout: {capture.err}")
        pytest.fail(f"CLI produced no output. exit_code={exit_code}")

    assert exit_code == 0, f"CLI failed with: {capture.out} {capture.err}"
    payload = json.loads(capture.out)
    assert payload["statistics"]["parse.state"] == ProcessingState.PARSED_TEXT.value
    assert (
        payload["statistics"]["assets.state"] == ProcessingState.ASSETS_EXTRACTED.value
    )
    stored_after_parse = context.repository.get(
        document.ref.tenant_id,
        document.ref.document_id,
        document.ref.version,
        workflow_id=document.ref.workflow_id,
    )
    assert stored_after_parse is not None
    assert len(stored_after_parse.assets) == 1
    assert parser.calls == 1

    caption_args = common_args + [
        "caption",
        "--tenant",
        document.ref.tenant_id,
        "--doc-id",
        str(document.ref.document_id),
        "--workflow-id",
        document.ref.workflow_id,
    ]
    exit_code = main(caption_args, context=context)
    capture = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(capture.out)
    assert payload["statistics"]["caption.state"] == ProcessingState.CAPTIONED.value
    stored_after_caption = context.repository.get(
        document.ref.tenant_id,
        document.ref.document_id,
        document.ref.version,
        workflow_id=document.ref.workflow_id,
    )
    assert stored_after_caption is not None
    captioned_asset = stored_after_caption.assets[0]
    assert captioned_asset.caption_method == "vlm_caption"
    assert captioned_asset.caption_source == "vlm"
    assert captioned_asset.caption_confidence is not None
    payload_bytes = context.storage.get(captioned_asset.blob.uri)
    expected_caption = context.captioner.caption(payload_bytes, None)
    assert captioned_asset.caption_confidence == pytest.approx(
        expected_caption["confidence"]
    )

    chunk_args = common_args + [
        "chunk",
        "--tenant",
        document.ref.tenant_id,
        "--doc-id",
        str(document.ref.document_id),
        "--workflow-id",
        document.ref.workflow_id,
    ]
    exit_code = main(chunk_args, context=context)
    capture = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(capture.out)
    assert payload["statistics"]["chunk.state"] == ProcessingState.CHUNKED.value
    assert payload["chunk_count"] == 1
    assert payload["preview"][0]["chunk_id"]
    assert payload["preview"][0]["parent_ref"]
    assert parser.calls >= 2
