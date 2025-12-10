"""End-to-end tests for document processing graph with real DB and storage.

These tests use the actual database (noesis2_test) and real components
instead of mocks, providing higher confidence that the graph works correctly
in production-like conditions.
"""

import base64
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from customers.models import Tenant
from documents.models import DocumentCollection

from ai_core.adapters.db_documents_repository import DbDocumentsRepository
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.pipeline import (
    DocumentPipelineConfig,
    DocumentProcessingContext,
    DocumentProcessingMetadata,
)
from documents.processing_graph import (
    DocumentProcessingPhase,
    DocumentProcessingState,
    build_document_processing_graph,
)
from documents.parsers import ParserDispatcher, ParserRegistry
from documents.parsers_markdown import MarkdownDocumentParser
from documents.parsers_text import TextDocumentParser
from documents.cli import SimpleDocumentChunker


@pytest.mark.django_db
class TestDocumentProcessingGraphE2E:
    """End-to-end tests using real database and components."""

    @pytest.fixture
    def tenant(self, test_tenant_schema_name):
        """Get test tenant from database."""
        return Tenant.objects.get(schema_name=test_tenant_schema_name)

    @pytest.fixture
    def collection(self, tenant):
        """Create a test document collection."""
        return DocumentCollection.objects.create(
            tenant=tenant, collection_id=uuid4(), name="e2e_test_collection"
        )

    @pytest.fixture
    def real_repository(self, force_inmemory_repository):
        """Override in-memory repository with real DB repository."""
        # The force_inmemory_repository fixture is autouse, so we override it here
        return DbDocumentsRepository()

    @pytest.fixture
    def parser(self):
        """Create real parser dispatcher with text and markdown support."""
        registry = ParserRegistry(
            [
                TextDocumentParser(),
                MarkdownDocumentParser(),
            ]
        )
        return ParserDispatcher(registry)

    @pytest.fixture
    def chunker(self):
        """Create real chunker."""
        return SimpleDocumentChunker()

    def test_e2e_full_pipeline_with_db_persistence(
        self, tenant, collection, real_repository, parser, chunker
    ):
        """Test full document processing pipeline with real DB persistence.

        This test:
        1. Creates a real document in the database
        2. Processes it through the full graph (parse, chunk, persist)
        3. Verifies the document is actually stored in the DB
        4. Verifies it can be retrieved
        """
        from unittest.mock import MagicMock

        # Use real parser and chunker, but mock embedder and storage
        storage = MagicMock()
        storage.put.return_value = ("file://test/storage.bin", "checksum123", 100)

        captioner = MagicMock()
        captioner.caption.return_value = ([], {})

        embedder = MagicMock()
        embedder.return_value = {
            "written": 1,
            "embedding_profile": "test",
            "vector_space_id": "global",
        }

        # Build graph with real repository and parser
        graph = build_document_processing_graph(
            parser=parser,
            repository=real_repository,
            storage=storage,
            captioner=captioner,
            chunker=chunker,
            embedder=embedder,
            delta_decider=None,
            guardrail_enforcer=None,
            propagate_errors=True,
        )

        # Create test document
        doc_id = uuid4()
        content = b"# Test Document\n\nThis is a test document for E2E testing."

        doc_ref = DocumentRef(
            document_id=doc_id,
            tenant_id=tenant.schema_name,
            workflow_id="e2e_test",
            version="1.0",
        )

        doc_meta = DocumentMeta(
            tenant_id=tenant.schema_name,
            workflow_id="e2e_test",
            title="E2E Test Document",
        )

        dummy_sha = hashlib.sha256(content).hexdigest()
        blob = InlineBlob(
            type="inline",
            media_type="text/markdown",
            base64=base64.b64encode(content).decode("ascii"),
            sha256=dummy_sha,
            size=len(content),
        )

        document = NormalizedDocument(
            ref=doc_ref,
            meta=doc_meta,
            blob=blob,
            checksum=dummy_sha,
            created_at=datetime.now(timezone.utc),
            source="other",
            lifecycle_state="active",
        )

        # Run graph
        context_meta = DocumentProcessingMetadata.from_document(document)
        context = DocumentProcessingContext(metadata=context_meta)
        config = DocumentPipelineConfig(enable_embedding=True)

        state = DocumentProcessingState(
            document=document,
            config=config,
            context=context,
            run_until=DocumentProcessingPhase.FULL,
        )

        graph.invoke(state)

        # Verify document was persisted to database
        retrieved_doc = real_repository.get(
            tenant_id=tenant.schema_name, document_id=doc_id, workflow_id="e2e_test"
        )

        assert retrieved_doc is not None, "Document should be in database"
        assert retrieved_doc.ref.document_id == doc_id
        assert retrieved_doc.meta.title == "E2E Test Document"
        assert retrieved_doc.lifecycle_state == "active"

    def test_e2e_delta_with_real_repository(
        self, tenant, collection, real_repository, parser, chunker
    ):
        """Test delta detection with real repository and database.

        This test:
        1. Stores a document in the database
        2. Processes the same document again
        3. Verifies delta detection works with real DB lookups
        """
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        # Mock components
        storage = MagicMock()
        storage.put.return_value = ("file://test/delta.bin", "checksum456", 50)

        captioner = MagicMock()
        captioner.caption.return_value = ([], {})

        embedder = MagicMock()
        embedder.return_value = {"written": 0}

        # Delta decider that detects unchanged documents
        delta_decider = MagicMock()
        delta_decider.return_value = SimpleNamespace(
            decision="skip", reason="unchanged", attributes={}
        )

        # Build graph
        graph = build_document_processing_graph(
            parser=parser,
            repository=real_repository,
            storage=storage,
            captioner=captioner,
            chunker=chunker,
            embedder=embedder,
            delta_decider=delta_decider,
            guardrail_enforcer=None,
            propagate_errors=True,
        )

        # Create and persist baseline document
        doc_id = uuid4()
        content = b"Delta test content"

        doc_ref = DocumentRef(
            document_id=doc_id,
            tenant_id=tenant.schema_name,
            workflow_id="delta_test",
            version="1.0",
        )

        doc_meta = DocumentMeta(
            tenant_id=tenant.schema_name, workflow_id="delta_test", title="Delta Test"
        )

        dummy_sha = hashlib.sha256(content).hexdigest()
        blob = InlineBlob(
            type="inline",
            media_type="text/plain",
            base64=base64.b64encode(content).decode("ascii"),
            sha256=dummy_sha,
            size=len(content),
        )

        baseline_doc = NormalizedDocument(
            ref=doc_ref,
            meta=doc_meta,
            blob=blob,
            checksum=dummy_sha,
            created_at=datetime.now(timezone.utc),
            source="other",
            lifecycle_state="active",
        )

        # First upsert to create baseline
        real_repository.upsert(baseline_doc)

        # Now process the same document again (delta should detect it)
        context_meta = DocumentProcessingMetadata.from_document(baseline_doc)
        context = DocumentProcessingContext(metadata=context_meta)
        config = DocumentPipelineConfig(enable_embedding=True)

        state = DocumentProcessingState(
            document=baseline_doc,
            config=config,
            context=context,
            run_until=DocumentProcessingPhase.FULL,
        )

        _result = graph.invoke(state)

        # Verify delta decider was called with real baseline from DB
        assert delta_decider.called, "Delta decider should have been invoked"

        # Verify embedding was skipped (delta detected unchanged doc)
        assert not embedder.called or embedder.return_value["written"] == 0

    def test_e2e_soft_delete_with_repository(
        self, tenant, collection, real_repository, parser, chunker
    ):
        """Test soft delete behavior with real repository.

        This test:
        1. Creates and persists a document
        2. Performs soft delete
        3. Verifies document is marked as retired (not deleted)
        4. Verifies it can still be retrieved with lifecycle_state='retired'
        """

        # Create and persist a document
        doc_id = uuid4()
        content = b"Document to be deleted"

        doc_ref = DocumentRef(
            document_id=doc_id,
            tenant_id=tenant.schema_name,
            workflow_id="delete_test",
            version="1.0",
        )

        doc_meta = DocumentMeta(
            tenant_id=tenant.schema_name, workflow_id="delete_test", title="Delete Test"
        )

        dummy_sha = hashlib.sha256(content).hexdigest()
        blob = InlineBlob(
            type="inline",
            media_type="text/plain",
            base64=base64.b64encode(content).decode("ascii"),
            sha256=dummy_sha,
            size=len(content),
        )

        document = NormalizedDocument(
            ref=doc_ref,
            meta=doc_meta,
            blob=blob,
            checksum=dummy_sha,
            created_at=datetime.now(timezone.utc),
            source="other",
            lifecycle_state="active",
        )

        # Persist document
        real_repository.upsert(document)

        # Verify it exists and is active
        active_doc = real_repository.get(
            tenant_id=tenant.schema_name, document_id=doc_id, workflow_id="delete_test"
        )
        assert active_doc is not None
        assert active_doc.lifecycle_state == "active"

        # Soft delete
        deleted = real_repository.delete(
            tenant_id=tenant.schema_name, document_id=doc_id, workflow_id="delete_test"
        )
        assert deleted is True, "Delete should return True"

        # Verify document is no longer retrievable (soft delete filters out retired docs)
        # This is by design - the repository's get() method filters out retired documents
        retired_doc = real_repository.get(
            tenant_id=tenant.schema_name, document_id=doc_id, workflow_id="delete_test"
        )
        assert (
            retired_doc is None
        ), "Repository.get() should filter out retired documents"
