from __future__ import annotations

import os
import re
import uuid
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

from django.conf import settings
from django.utils import timezone

from ai_core.segmentation import segment_markdown_blocks
from ai_core.tool_contracts.base import tool_context_from_meta
from documents.pipeline import (
    DocumentProcessingContext,
    DocumentProcessingMetadata,
    ParsedTextBlock,
)


def _split_sentences_impl(text: str) -> List[str]:
    """Best-effort sentence segmentation that retains punctuation."""

    pattern = re.compile(r"[^.!?]+(?:[.!?]+|\Z)")
    sentences = [segment.strip() for segment in pattern.findall(text)]
    sentences = [s for s in sentences if s]
    if sentences:
        if len(sentences) == 1 and not re.search(r"[.!?]", text):
            sentences = []
        else:
            return sentences
    # Fallback: use paragraphs or lines if no sentence boundary detected
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    return parts or [text.strip()]


if tiktoken:
    try:
        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - defensive fallback
        _TOKEN_ENCODING = None
else:
    _TOKEN_ENCODING = None


_FORCE_WHITESPACE_TOKENIZER = os.getenv(
    "AI_CORE_FORCE_WHITESPACE_TOKENIZER", ""
).lower() in (
    "1",
    "true",
    "yes",
)

_PRONOUN_PATTERN = re.compile(
    r"\b(ich|mich|mir|du|dich|dir|er|ihn|ihm|sie|ihr|es|wir|uns|euch|"
    r"they|them|their|theirs|he|him|his|she|her|hers|we|us|our|ours|i|me|my|mine)\b",
    re.IGNORECASE,
)

_LIST_LIKE_KEYWORDS = (
    "faq",
    "liste",
    "list",
    "bullet",
    "checklist",
    "glossary",
    "table",
)

_NARRATIVE_KEYWORDS = (
    "narrative",
    "narrativ",
    "story",
    "bericht",
    "report",
    "fallstudie",
    "memo",
    "conversation",
    "transkript",
)


def _should_use_tiktoken() -> bool:
    return _TOKEN_ENCODING is not None and not _FORCE_WHITESPACE_TOKENIZER


def set_tokenizer_override(force_whitespace: bool) -> None:
    global _FORCE_WHITESPACE_TOKENIZER
    _FORCE_WHITESPACE_TOKENIZER = force_whitespace


@contextmanager
def force_whitespace_tokenizer() -> Iterator[None]:
    """Temporarily force the whitespace-based tokenizer fallback."""

    previous = _FORCE_WHITESPACE_TOKENIZER
    set_tokenizer_override(True)
    try:
        yield
    finally:
        set_tokenizer_override(previous)


def _token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    if _should_use_tiktoken():
        # `disallowed_special=()` ensures consistent behaviour across tiktoken versions.
        return max(1, len(_TOKEN_ENCODING.encode(stripped, disallowed_special=())))

    whitespace_tokens = [segment for segment in stripped.split() if segment]
    if len(whitespace_tokens) > 1:
        return len(whitespace_tokens)

    return max(1, len(stripped))


def _split_by_limit(text: str, hard_limit: int) -> List[str]:
    if not text:
        return []

    limit = max(1, int(hard_limit))

    if _should_use_tiktoken():
        token_ids = _TOKEN_ENCODING.encode(text, disallowed_special=())
        if not token_ids:
            return []
        if len(token_ids) <= limit:
            return [text]
        parts: List[str] = []
        for start in range(0, len(token_ids), limit):
            chunk_ids = token_ids[start : start + limit]
            parts.append(_TOKEN_ENCODING.decode(chunk_ids))
        return parts

    whitespace_chunks = list(re.finditer(r"\S+\s*", text))
    if len(whitespace_chunks) > 1:
        if len(whitespace_chunks) <= limit:
            return [text]
        parts: List[str] = []
        current_segments: List[str] = []
        current_tokens = 0

        for match in whitespace_chunks:
            segment = match.group(0)
            stripped = segment.strip()

            if not stripped:
                continue

            if current_tokens + 1 > limit and current_segments:
                parts.append("".join(current_segments).rstrip())
                current_segments = []
                current_tokens = 0

            current_segments.append(segment)
            current_tokens += 1

        if current_segments:
            parts.append("".join(current_segments).rstrip())

        return [part for part in parts if part]

    if len(text) <= limit:
        return [text]

    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _estimate_overlap_ratio(text: str, meta: Dict[str, str]) -> float:
    """Estimate chunk overlap ratio between 10% and 25%."""

    ratio_min = 0.10
    ratio_max = 0.25
    stripped = text.strip()
    if not stripped:
        return ratio_min

    ratio = 0.15
    context = tool_context_from_meta(meta)
    doc_type = str(
        meta.get("doc_class")
        or context.business.collection_id
        or meta.get("document_type")
        or meta.get("type")
        or ""
    ).lower()
    if doc_type:
        if any(keyword in doc_type for keyword in _LIST_LIKE_KEYWORDS):
            return ratio_min
        if any(keyword in doc_type for keyword in _NARRATIVE_KEYWORDS):
            ratio = max(ratio, 0.22)

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        bullet_lines = sum(
            1 for line in lines if re.match(r"^(?:[-*ƒ?½]\\s|\\d+\\.\\s)", line)
        )
        bullet_ratio = bullet_lines / max(1, len(lines))
        if bullet_ratio >= 0.4:
            ratio -= 0.03
        elif bullet_ratio <= 0.1 and len(lines) > 4:
            ratio += 0.02

    words = re.findall(r"\\b\\w+\\b", stripped)
    word_count = len(words)
    if word_count:
        pronoun_count = len(_PRONOUN_PATTERN.findall(stripped))
        pronoun_ratio = pronoun_count / word_count
        if pronoun_ratio >= 0.07:
            ratio += 0.07
        elif pronoun_ratio >= 0.04:
            ratio += 0.04
        elif pronoun_ratio <= 0.015:
            ratio -= 0.02

    return max(ratio_min, min(ratio, ratio_max))


def _resolve_overlap_tokens(
    text: str,
    meta: Dict[str, str],
    *,
    target_tokens: int,
    hard_limit: int,
) -> int:
    configured_limit = getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", None)
    try:
        configured_value = (
            int(configured_limit) if configured_limit is not None else None
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive
        configured_value = None

    if configured_value is not None and configured_value <= 0:
        return 0

    ratio = _estimate_overlap_ratio(text, meta)
    overlap = int(round(target_tokens * ratio))
    if ratio > 0 and overlap == 0:
        overlap = 1

    if configured_value is not None:
        overlap = min(overlap, configured_value)

    overlap = min(overlap, max(0, target_tokens - 1))

    return max(0, min(overlap, hard_limit))


def _chunkify(
    sentences: Sequence[str],
    *,
    target_tokens: int,
    overlap_tokens: int,
    hard_limit: int,
) -> List[str]:
    chunks: List[str] = []
    current: List[Tuple[str, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunk_text = " ".join(sentence for sentence, _ in current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if overlap_tokens > 0:
            retained: List[Tuple[str, int]] = []
            retained_tokens = 0
            for sentence, tokens in reversed(current):
                retained.insert(0, (sentence, tokens))
                retained_tokens += tokens
                if retained_tokens >= overlap_tokens:
                    break
            current = retained
            current_tokens = retained_tokens
        else:
            current = []
            current_tokens = 0

    for sentence in sentences:
        tokens = _token_count(sentence)
        if tokens > hard_limit:
            sub_sentences = _split_by_limit(sentence, hard_limit)
            for sub_sentence in sub_sentences:
                sub_tokens = _token_count(sub_sentence)
                if sub_tokens > hard_limit:
                    # Guard against pathological tokenizer fallbacks by forcing a hard trim.
                    sub_sentence = sub_sentence[:hard_limit]
                    sub_tokens = _token_count(sub_sentence)
                if current_tokens + sub_tokens > hard_limit:
                    flush()
                current.append((sub_sentence, sub_tokens))
                current_tokens += sub_tokens
                if current_tokens >= target_tokens:
                    flush()
            continue

        if current_tokens + tokens > hard_limit and current:
            flush()

        current.append((sentence, tokens))
        current_tokens += tokens

        if current_tokens >= target_tokens:
            flush()

    flush()
    return chunks


def _build_chunk_prefix(meta: Dict[str, str]) -> str:
    parts: List[str] = []
    breadcrumbs = meta.get("breadcrumbs")
    if isinstance(breadcrumbs, Iterable) and not isinstance(breadcrumbs, (str, bytes)):
        crumb_parts = [str(item).strip() for item in breadcrumbs if str(item).strip()]
        if crumb_parts:
            parts.append(" / ".join(crumb_parts))
    title = meta.get("title")
    if title:
        parts.append(str(title).strip())
    return " — ".join(part for part in parts if part)


def _resolve_parent_capture_max_depth() -> int:
    try:
        value = getattr(settings, "RAG_PARENT_CAPTURE_MAX_DEPTH", 0)
    except Exception:
        return 0

    try:
        depth = int(value)
    except (TypeError, ValueError):
        return 0

    return depth if depth > 0 else 0


def _resolve_parent_capture_max_bytes() -> int:
    try:
        value = getattr(settings, "RAG_PARENT_CAPTURE_MAX_BYTES", 0)
    except Exception:
        return 0

    try:
        byte_limit = int(value)
    except (TypeError, ValueError):
        return 0

    return byte_limit if byte_limit > 0 else 0


_PARSED_BLOCK_KINDS = {
    "paragraph",
    "heading",
    "list",
    "table_summary",
    "slide",
    "note",
    "code",
    "other",
}


def _coerce_block_kind(value: object) -> str:
    if value is None:
        return "paragraph"
    candidate = str(value).strip().lower()
    return candidate if candidate in _PARSED_BLOCK_KINDS else "paragraph"


def _coerce_section_path(value: object) -> Optional[Tuple[str, ...]]:
    if isinstance(value, (list, tuple)):
        path = tuple(str(part).strip() for part in value if str(part).strip())
        return path or None
    return None


def _build_parsed_blocks(
    *,
    text: str,
    structured_blocks: Sequence[Mapping[str, object]],
    mask_fn: Callable[[str], str],
) -> List[ParsedTextBlock]:
    blocks: List[ParsedTextBlock] = []
    if structured_blocks:
        for block in structured_blocks:
            text_value = block.get("text") if isinstance(block, Mapping) else None
            if text_value is None:
                continue
            text_str = str(text_value).strip()
            if not text_str:
                continue
            kind = _coerce_block_kind(block.get("kind"))
            section_path = _coerce_section_path(block.get("section_path"))
            page_index = None
            raw_page = block.get("page_index")
            if raw_page is not None:
                try:
                    page_index = int(raw_page)
                except (TypeError, ValueError):
                    page_index = None
            table_meta = None
            raw_table = block.get("table_meta")
            if isinstance(raw_table, Mapping):
                table_meta = raw_table
            language = None
            raw_language = block.get("language")
            if isinstance(raw_language, str) and raw_language.strip():
                language = raw_language.strip()
            try:
                blocks.append(
                    ParsedTextBlock(
                        text=mask_fn(text_str),
                        kind=kind,
                        section_path=section_path,
                        page_index=page_index,
                        table_meta=table_meta,
                        language=language,
                    )
                )
            except ValueError:
                continue
        return blocks

    heading_pattern = re.compile(r"^\\s{0,3}(#{1,6})\\s+(.*)$")
    current_path: List[str] = []
    for segment in segment_markdown_blocks(text):
        stripped = segment.strip()
        if not stripped:
            continue
        heading_match = heading_pattern.match(stripped)
        if heading_match:
            hashes, heading_title = heading_match.groups()
            level = len(hashes)
            title = heading_title.strip()
            if not title:
                continue
            while len(current_path) >= level:
                current_path.pop()
            current_path.append(title)
            try:
                blocks.append(
                    ParsedTextBlock(
                        text=mask_fn(title),
                        kind="heading",
                        section_path=tuple(current_path),
                    )
                )
            except ValueError:
                continue
            continue
        try:
            blocks.append(
                ParsedTextBlock(
                    text=mask_fn(stripped),
                    kind="paragraph",
                    section_path=tuple(current_path) if current_path else None,
                )
            )
        except ValueError:
            continue

    if not blocks and text.strip():
        try:
            blocks.append(ParsedTextBlock(text=mask_fn(text.strip()), kind="paragraph"))
        except ValueError:
            pass
    return blocks


def _build_processing_context(
    *,
    meta: Mapping[str, object],
    tool_context: Any,
) -> Optional[DocumentProcessingContext]:
    workflow_id = meta.get("workflow_id") or getattr(
        tool_context.business, "workflow_id", None
    )
    if workflow_id is None or str(workflow_id).strip() == "":
        return None
    document_id = meta.get("document_id") or getattr(
        tool_context.business, "document_id", None
    )
    if document_id is None or str(document_id).strip() == "":
        return None
    try:
        document_uuid = uuid.UUID(str(document_id))
    except (TypeError, ValueError, AttributeError):
        return None

    metadata = DocumentProcessingMetadata(
        tenant_id=str(tool_context.scope.tenant_id),
        collection_id=getattr(tool_context.business, "collection_id", None),
        case_id=getattr(tool_context.business, "case_id", None),
        workflow_id=str(workflow_id),
        document_id=document_uuid,
        source=str(meta.get("source")) if meta.get("source") else None,
        created_at=timezone.now(),
        trace_id=getattr(tool_context.scope, "trace_id", None),
    )
    return DocumentProcessingContext(
        metadata=metadata,
        trace_id=metadata.trace_id,
        span_id=metadata.span_id,
    )
