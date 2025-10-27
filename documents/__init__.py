"""Convenience exports for document ingestion configuration utilities."""

from .pipeline import (  # noqa: F401
    DocumentChunkArtifact,
    DocumentChunker,
    DocumentComponents,
    DocumentContracts,
    DocumentParseArtifact,
    DocumentPipelineConfig,
    DocumentProcessingOrchestrator,
    DocumentProcessingContext,
    DocumentProcessingMetadata,
    DocumentProcessingOutcome,
    ProcessingState,
    persist_parsed_document,
    require_document_components,
    require_document_contracts,
)
from .parsers import (  # noqa: F401
    DocumentParser,
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    ParsedAsset,
    ParsedEntity,
    ParsedResult,
    ParsedTextBlock,
    ParserDispatcher,
    ParserRegistry,
    compute_parser_stats,
    normalize_diagnostics,
)
from .parsers_docx import DocxDocumentParser  # noqa: F401
from .parsers_html import HtmlDocumentParser  # noqa: F401
from .parsers_markdown import MarkdownDocumentParser  # noqa: F401
from .parsers_pdf import PdfDocumentParser  # noqa: F401
from .parsers_pptx import PptxDocumentParser  # noqa: F401
from .providers import (  # noqa: F401
    ProviderReference,
    build_external_reference,
    parse_provider_reference,
)

__all__ = [
    "DocumentChunkArtifact",
    "DocumentChunker",
    "DocumentComponents",
    "DocumentContracts",
    "DocumentParseArtifact",
    "DocumentPipelineConfig",
    "DocumentProcessingOrchestrator",
    "DocumentProcessingContext",
    "DocumentProcessingMetadata",
    "DocumentProcessingOutcome",
    "ProcessingState",
    "persist_parsed_document",
    "require_document_components",
    "require_document_contracts",
    "DocumentParser",
    "ParseResult",
    "ParseStatus",
    "ParserContent",
    "ParserStats",
    "ParsedAsset",
    "ParsedEntity",
    "ParsedResult",
    "ParsedTextBlock",
    "ParserDispatcher",
    "ParserRegistry",
    "compute_parser_stats",
    "normalize_diagnostics",
    "DocxDocumentParser",
    "HtmlDocumentParser",
    "MarkdownDocumentParser",
    "PdfDocumentParser",
    "PptxDocumentParser",
    "ProviderReference",
    "build_external_reference",
    "parse_provider_reference",
]
