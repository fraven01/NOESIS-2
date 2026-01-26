from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from django.conf import settings

from common.logging import get_logger

from ai_core.infra.prompts import load as load_prompt
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ContextualEnrichmentConfig:
    enabled: bool
    model_label: str
    max_document_chars: int
    max_chunk_chars: int
    max_chunks: int
    max_prefix_chars: int
    max_prefix_words: int


def get_contextual_enrichment_config() -> ContextualEnrichmentConfig:
    enabled = bool(getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT", False))
    model_label = str(
        getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT_MODEL", "fast")
    ).strip()
    max_document_chars = int(
        getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT_MAX_DOC_CHARS", 12000)
    )
    max_chunk_chars = int(
        getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT_MAX_CHUNK_CHARS", 2000)
    )
    max_chunks = int(getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT_MAX_CHUNKS", 120))
    max_prefix_chars = int(
        getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT_MAX_PREFIX_CHARS", 800)
    )
    max_prefix_words = int(
        getattr(settings, "RAG_CONTEXTUAL_ENRICHMENT_MAX_PREFIX_WORDS", 120)
    )
    return ContextualEnrichmentConfig(
        enabled=enabled,
        model_label=model_label or "fast",
        max_document_chars=max(0, max_document_chars),
        max_chunk_chars=max(0, max_chunk_chars),
        max_chunks=max(0, max_chunks),
        max_prefix_chars=max(0, max_prefix_chars),
        max_prefix_words=max(0, max_prefix_words),
    )


def resolve_contextual_enrichment_config(
    enabled: bool | None,
) -> ContextualEnrichmentConfig:
    base = get_contextual_enrichment_config()
    if enabled is None or enabled is base.enabled:
        return base
    return ContextualEnrichmentConfig(
        enabled=enabled,
        model_label=base.model_label,
        max_document_chars=base.max_document_chars,
        max_chunk_chars=base.max_chunk_chars,
        max_chunks=base.max_chunks,
        max_prefix_chars=base.max_prefix_chars,
        max_prefix_words=base.max_prefix_words,
    )


def generate_contextual_prefixes(
    document_text: str,
    chunk_entries: Sequence[Mapping[str, Any]],
    context: Any,
    config: ContextualEnrichmentConfig,
) -> list[str | None]:
    prefixes: list[str | None] = [None] * len(chunk_entries)
    if not config.enabled:
        return prefixes

    doc_text = _trim_text(document_text, config.max_document_chars)
    if not doc_text.strip():
        return prefixes

    limit = config.max_chunks
    for index, entry in enumerate(chunk_entries):
        if limit > 0 and index >= limit:
            break
        chunk_text = str(entry.get("text") or "").strip()
        if not chunk_text:
            continue
        chunk_text = _trim_text(chunk_text, config.max_chunk_chars)
        prompt_text, prompt_version = _build_contextual_prompt(
            document_text=doc_text,
            chunk_text=chunk_text,
        )
        metadata = _resolve_context_metadata(context, prompt_version=prompt_version)
        try:
            response = llm_client.call(config.model_label, prompt_text, metadata)
        except (LlmClientError, RateLimitError) as exc:
            logger.warning(
                "ingestion.contextual_enrichment.failed",
                extra={
                    "error": type(exc).__name__,
                    "error_message": str(exc),
                    "chunk_index": index,
                },
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "ingestion.contextual_enrichment.failed",
                extra={
                    "error": type(exc).__name__,
                    "error_message": str(exc),
                    "chunk_index": index,
                },
            )
            continue

        prefix = _extract_contextual_prefix(str(response.get("text") or ""))
        if not prefix:
            continue
        prefix = _cap_prefix(
            prefix,
            max_chars=config.max_prefix_chars,
            max_words=config.max_prefix_words,
        )
        if prefix:
            prefixes[index] = prefix

    return prefixes


def _resolve_context_metadata(
    context: Any,
    *,
    prompt_version: str,
) -> dict[str, object]:
    tenant_id = None
    case_id = None
    trace_id = None

    scope = getattr(context, "scope", None)
    if scope is not None:
        tenant_id = getattr(scope, "tenant_id", None)
        trace_id = getattr(scope, "trace_id", None)
    business = getattr(context, "business", None)
    if business is not None:
        case_id = getattr(business, "case_id", None)
    metadata = getattr(context, "metadata", None)
    if metadata is not None:
        tenant_id = tenant_id or getattr(metadata, "tenant_id", None)
        case_id = case_id or getattr(metadata, "case_id", None)
        trace_id = trace_id or getattr(metadata, "trace_id", None)

    trace_id = trace_id or getattr(context, "trace_id", None)

    return {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "trace_id": trace_id,
        "prompt_version": prompt_version,
    }


def _trim_text(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if max_chars <= 0:
        return cleaned
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _cap_prefix(text: str, *, max_chars: int, max_words: int) -> str:
    cleaned = " ".join((text or "").split())
    if max_words > 0:
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words])
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def _extract_contextual_prefix(text: str) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    cleaned = cleaned.strip().strip('"').strip("'")
    return cleaned or None


def _build_contextual_prompt(
    *,
    document_text: str,
    chunk_text: str,
) -> tuple[str, str]:
    prompt = load_prompt("retriever/contextual_enrichment")
    prompt_text = (
        f"{prompt['text']}\n\n<document>{document_text}</document>\n"
        f"<chunk>{chunk_text}</chunk>"
    )
    return prompt_text, str(prompt.get("version") or "")
