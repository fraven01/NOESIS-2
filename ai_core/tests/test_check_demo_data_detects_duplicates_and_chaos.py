"""AI-Core smoke scenario validating the in-memory documents CLI helpers."""

from __future__ import annotations

import base64
import json
from typing import Iterable
from uuid import UUID, uuid4

from documents.cli import CLIContext, main, SimpleDocumentChunker
from documents.parsers import ParserRegistry, ParserDispatcher
from documents.parsers_markdown import MarkdownDocumentParser
from documents.parsers_html import HtmlDocumentParser
from documents.parsers_docx import DocxDocumentParser
from documents.parsers_pptx import PptxDocumentParser
from documents.parsers_pdf import PdfDocumentParser
from documents.captioning import DeterministicCaptioner
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage


def _context() -> CLIContext:
    storage = InMemoryStorage()
    repository = InMemoryDocumentsRepository(storage=storage)
    registry = ParserRegistry(
        [
            MarkdownDocumentParser(),
            HtmlDocumentParser(),
            DocxDocumentParser(),
            PptxDocumentParser(),
            PdfDocumentParser(),
        ]
    )
    dispatcher = ParserDispatcher(registry)
    captioner = DeterministicCaptioner()
    chunker = SimpleDocumentChunker()
    return CLIContext(
        repository=repository,
        storage=storage,
        parser_registry=registry,
        parser=dispatcher,
        captioner=captioner,
        chunker=chunker,
    )


def _run_cli(
    args: Iterable[str], context: CLIContext, *, capsys
) -> tuple[int, dict[str, object]]:
    exit_code = main(["--json", *args], context=context)
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out.strip()) if captured.out else {}
    return exit_code, payload


def _encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def test_documents_cli_roundtrip_uses_inmemory_repository(capsys):
    context = _context()
    tenant_id = "tenant-ai-core"
    collection_id = str(uuid4())

    add_code, doc_payload = _run_cli(
        [
            "docs",
            "add",
            "--tenant",
            tenant_id,
            "--workflow-id",
            "workflow-ai-core",
            "--collection",
            collection_id,
            "--title",
            "AI Core Demo",
            "--inline",
            _encode(b"document-body"),
            "--media-type",
            "text/plain",
            "--source",
            "upload",
        ],
        context,
        capsys=capsys,
    )
    assert add_code == 0

    document_id = doc_payload["ref"]["document_id"]

    asset_code, asset_payload = _run_cli(
        [
            "assets",
            "add",
            "--tenant",
            tenant_id,
            "--workflow-id",
            "workflow-ai-core",
            "--document",
            document_id,
            "--media-type",
            "image/png",
            "--inline",
            _encode(b"asset-bytes"),
            "--caption-method",
            "manual",
        ],
        context,
        capsys=capsys,
    )
    assert asset_code == 0

    asset_id = asset_payload["ref"]["asset_id"]

    stored = context.repository.get(
        tenant_id,
        UUID(document_id),
        prefer_latest=True,
        workflow_id="workflow-ai-core",
    )
    assert stored is not None
    assert stored.ref.tenant_id == tenant_id
    assert stored.ref.collection_id == UUID(collection_id)
    assert stored.checksum == doc_payload["checksum"]
    assert len(stored.assets) == 1
    assert str(stored.assets[0].ref.asset_id) == asset_id

    asset_refs, cursor = context.repository.list_assets_by_document(
        tenant_id, UUID(document_id), workflow_id="workflow-ai-core"
    )
    assert cursor is None
    assert len(asset_refs) == 1
    assert str(asset_refs[0].asset_id) == asset_id
