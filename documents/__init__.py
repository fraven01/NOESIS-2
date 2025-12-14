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
from .processing_graph import DocumentProcessingPhase  # noqa: F401
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
from .normalization import (  # noqa: F401
    document_payload_bytes,
    normalized_primary_text,
)
from .parsers_docx import DocxDocumentParser  # noqa: F401
from .parsers_html import HtmlDocumentParser  # noqa: F401
from .parsers_image import ImageDocumentParser  # noqa: F401
from .parsers_markdown import MarkdownDocumentParser  # noqa: F401
from .parsers_pdf import PdfDocumentParser  # noqa: F401
from .parsers_pptx import PptxDocumentParser  # noqa: F401
from .parsers_text import TextDocumentParser  # noqa: F401
from .parsers_xlsx import XlsxDocumentParser  # noqa: F401
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
    "DocumentProcessingPhase",
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
    "document_payload_bytes",
    "normalized_primary_text",
    "DocxDocumentParser",
    "HtmlDocumentParser",
    "ImageDocumentParser",
    "MarkdownDocumentParser",
    "PdfDocumentParser",
    "PptxDocumentParser",
    "TextDocumentParser",
    "XlsxDocumentParser",
    "ProviderReference",
    "build_external_reference",
    "parse_provider_reference",
]
