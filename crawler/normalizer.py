"""Crawler-specific wrapper around shared document normalisation."""

from __future__ import annotations

from typing import Optional, Sequence
from uuid import UUID

from documents import normalize_diagnostics
from documents.contracts import NormalizedDocument as ContractsNormalizedDocument
from documents.normalization import (
    document_parser_stats,
    document_payload_bytes,
    normalize_from_parse,
    normalized_primary_text,
    resolve_provider_reference,
)
from documents.parsers import ParseResult

from .contracts import NormalizedSource


def build_normalized_document(
    *,
    parse_result: ParseResult,
    source: NormalizedSource,
    tenant_id: str,
    workflow_id: str,
    document_id: UUID | str,
    tags: Optional[Sequence[str]] = None,
) -> ContractsNormalizedDocument:
    """Compose a :class:`documents.contracts.NormalizedDocument` instance."""

    return normalize_from_parse(
        parse_result=parse_result,
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        canonical_source=source.canonical_source,
        provider=source.provider,
        external_id=source.external_id,
        provider_tags=source.provider_tags,
        tags=tags,
        fetch_result=parse_result.fetch,
        ingest_source="crawler",
    )


__all__ = [
    "build_normalized_document",
    "document_parser_stats",
    "document_payload_bytes",
    "normalize_diagnostics",
    "normalized_primary_text",
    "resolve_provider_reference",
]

