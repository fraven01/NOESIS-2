from __future__ import annotations

from collections.abc import Mapping
import html
import re


def coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_optional_float(
    value: object, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and candidate < minimum:
        candidate = minimum
    if maximum is not None and candidate > maximum:
        candidate = maximum
    return candidate


def coerce_optional_int(value: object, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and candidate < minimum:
        candidate = minimum
    return candidate


def build_hybrid_config_from_payload(
    payload: Mapping[str, object] | None,
) -> dict[str, float | int | None]:
    hybrid: dict[str, float | int | None] = {
        "alpha": 0.5,
        "top_k": 5,
        "min_sim": 0.0,
    }
    if not isinstance(payload, Mapping):
        payload = None
    overrides = {
        "alpha": coerce_optional_float(
            payload.get("alpha") if payload is not None else None,
            minimum=0.0,
            maximum=1.0,
        ),
        "min_sim": coerce_optional_float(
            payload.get("min_sim") if payload is not None else None,
            minimum=0.0,
            maximum=1.0,
        ),
        "top_k": coerce_optional_int(
            payload.get("top_k") if payload is not None else None, minimum=1
        ),
        "vec_limit": coerce_optional_int(
            payload.get("vec_limit") if payload is not None else None, minimum=1
        ),
        "lex_limit": coerce_optional_int(
            payload.get("lex_limit") if payload is not None else None, minimum=1
        ),
        "trgm_limit": coerce_optional_float(
            payload.get("trgm_limit") if payload is not None else None,
            minimum=0.0,
            maximum=1.0,
        ),
        "max_candidates": coerce_optional_int(
            payload.get("max_candidates") if payload is not None else None, minimum=1
        ),
        "diversify_strength": coerce_optional_float(
            payload.get("diversify_strength") if payload is not None else None,
            minimum=0.0,
            maximum=1.0,
        ),
    }
    for key, value in overrides.items():
        if value is not None:
            hybrid[key] = value
    return hybrid


def build_hybrid_config(request) -> dict[str, float | int | None]:
    return build_hybrid_config_from_payload(getattr(request, "POST", None))


def _resolve_citation_label(snippet: Mapping[str, object]) -> str:
    raw_label = snippet.get("citation")
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()
    for key in ("source", "id"):
        candidate = snippet.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "Source"


_FILENAME_PATTERN = re.compile(r"[^\\/]+\.(pdf|docx|doc|txt|md|pptx|xlsx)\b", re.I)


def _basename_from_path(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    parts = re.split(r"[\\/]", cleaned)
    if not parts:
        return None
    candidate = parts[-1].strip()
    return candidate or None


def _extract_display_label(snippet: Mapping[str, object]) -> str | None:
    meta = snippet.get("meta")
    if isinstance(meta, Mapping):
        for key in (
            "title",
            "filename",
            "file_name",
            "source_title",
            "origin_uri",
            "url",
            "source_url",
        ):
            candidate = meta.get(key)
            if isinstance(candidate, str) and candidate.strip():
                if key in {"origin_uri", "url", "source_url"}:
                    base = _basename_from_path(candidate)
                    if base:
                        return base
                return candidate.strip()

    text = snippet.get("text")
    if isinstance(text, str):
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line and _FILENAME_PATTERN.search(first_line):
            return first_line.strip()

    return None


def build_snippet_items(
    snippets: list[dict],
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    if limit is None or limit <= 0:
        limit = 3
    for snippet in snippets[:limit]:
        document_id = snippet.get("document_id")
        if not isinstance(document_id, str):
            meta = snippet.get("meta")
            if isinstance(meta, dict):
                document_id = meta.get("document_id")
                chunk_id = meta.get("chunk_id")
            else:
                chunk_id = None
        else:
            meta = snippet.get("meta")
            if isinstance(meta, dict):
                chunk_id = meta.get("chunk_id")
            else:
                chunk_id = None
        download_url = _build_download_url(
            document_id if isinstance(document_id, str) else None
        )
        source = snippet.get("source") or "Unknown"
        display_label = _extract_display_label(snippet)
        text = snippet.get("text") or ""
        text_value = str(text) if text is not None else ""
        try:
            score_value = float(snippet.get("score", 0))
        except (TypeError, ValueError):
            score_value = 0.0
        items.append(
            {
                "source": source,
                "display_label": display_label or source,
                "title": text_value[:200],
                "text": text_value,
                "score_percent": int(score_value * 100),
                "download_url": download_url,
                "citation_label": _resolve_citation_label(snippet),
                "document_id": document_id,
                "chunk_id": chunk_id,
            }
        )
    return items


_SOURCE_LABEL_PATTERN = re.compile(r"^source\s+([a-z])$", re.I)
_SNIPPET_LABEL_PATTERN = re.compile(r"^snippet\s+(\d+)$", re.I)


def _build_snippet_label_maps(
    snippet_items: list[dict[str, object]] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    citation_map: dict[str, str] = {}
    source_map: dict[str, str] = {}
    source_conflicts: set[str] = set()
    if not snippet_items:
        return citation_map, source_map
    for item in snippet_items:
        if not isinstance(item, dict):
            continue
        display_label = item.get("display_label")
        if not isinstance(display_label, str) or not display_label.strip():
            continue
        citation_label = item.get("citation_label")
        if isinstance(citation_label, str) and citation_label.strip():
            citation_map[citation_label.strip()] = display_label.strip()
        source_label = item.get("source")
        if isinstance(source_label, str) and source_label.strip():
            key = source_label.strip()
            if key in source_map and source_map.get(key) != display_label.strip():
                source_conflicts.add(key)
            else:
                source_map[key] = display_label.strip()
    for key in source_conflicts:
        source_map.pop(key, None)
    return citation_map, source_map


def _resolve_used_source_label(
    label: object,
    *,
    index: int,
    snippet_items: list[dict[str, object]] | None,
    citation_map: dict[str, str],
    source_map: dict[str, str],
) -> str | None:
    if isinstance(label, str):
        cleaned = label.strip()
        if not cleaned:
            cleaned = ""
    else:
        cleaned = ""

    if cleaned:
        mapped = citation_map.get(cleaned) or source_map.get(cleaned)
        if mapped:
            return mapped

        match = _SOURCE_LABEL_PATTERN.match(cleaned)
        if match:
            offset = ord(match.group(1).lower()) - ord("a")
            if snippet_items and 0 <= offset < len(snippet_items):
                display_label = snippet_items[offset].get("display_label")
                if isinstance(display_label, str) and display_label.strip():
                    return display_label.strip()

        match = _SNIPPET_LABEL_PATTERN.match(cleaned)
        if match:
            try:
                offset = int(match.group(1)) - 1
            except (TypeError, ValueError):
                offset = -1
            if snippet_items and 0 <= offset < len(snippet_items):
                display_label = snippet_items[offset].get("display_label")
                if isinstance(display_label, str) and display_label.strip():
                    return display_label.strip()

    if snippet_items and 0 <= index < len(snippet_items):
        display_label = snippet_items[index].get("display_label")
        if isinstance(display_label, str) and display_label.strip():
            return display_label.strip()
    return cleaned or None


def build_used_source_items(
    used_sources: object,
    *,
    snippet_items: list[dict[str, object]] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    if not isinstance(used_sources, list):
        return []
    if limit is not None and limit > 0:
        sources = used_sources[:limit]
    else:
        sources = used_sources
    citation_map, source_map = _build_snippet_label_maps(snippet_items)
    items: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        label = source.get("label") or source.get("id") or "Source"
        display_label = _resolve_used_source_label(
            label,
            index=index,
            snippet_items=snippet_items,
            citation_map=citation_map,
            source_map=source_map,
        )
        try:
            relevance = float(source.get("relevance_score", 0))
        except (TypeError, ValueError):
            relevance = 0.0
        items.append(
            {
                "label": str(display_label or label),
                "score_percent": max(0, min(100, int(relevance * 100))),
                "id": source.get("id"),
            }
        )
    return items


def build_passage_items_for_workbench(snippets: object) -> list[dict[str, object]]:
    if not isinstance(snippets, list):
        return []

    try:
        from ai_core.rag.passage_assembly import assemble_passages
        from ai_core.rag.schemas import Chunk
    except Exception:
        return []

    chunks: list[Chunk] = []
    reference_map: dict[str, dict[str, list[str]]] = {}
    for index, snippet in enumerate(snippets):
        if not isinstance(snippet, dict):
            continue
        meta = snippet.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        chunk_id = meta.get("chunk_id") or snippet.get("id") or f"snippet-{index}"
        document_id = meta.get("document_id") or snippet.get("id")
        section_path = meta.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        chunk_index = meta.get("chunk_index")
        if chunk_index is None:
            chunk_index = index
        chunk_meta = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "section_path": section_path,
            "chunk_index": chunk_index,
            "score": snippet.get("score", 0.0),
        }
        parent_ids = meta.get("parent_ids")
        if parent_ids is not None:
            chunk_meta["parent_ids"] = parent_ids
        reference_ids = meta.get("reference_ids") or meta.get("references")
        if isinstance(reference_ids, list):
            cleaned_ids = [
                str(value).strip()
                for value in reference_ids
                if isinstance(value, (str, int, float))
            ]
        else:
            cleaned_ids = []
        reference_labels = meta.get("reference_labels")
        if isinstance(reference_labels, list):
            cleaned_labels = [
                str(value).strip()
                for value in reference_labels
                if isinstance(value, (str, int, float))
            ]
        else:
            cleaned_labels = []
        reference_map[str(chunk_id)] = {
            "reference_ids": [value for value in cleaned_ids if value],
            "reference_labels": [value for value in cleaned_labels if value],
        }
        chunks.append(Chunk(content=str(snippet.get("text") or ""), meta=chunk_meta))

    passages = assemble_passages(chunks)
    items: list[dict[str, object]] = []
    for passage in passages:
        passage_reference_ids: list[str] = []
        passage_reference_labels: list[str] = []
        for chunk_id in passage.chunk_ids:
            entry = reference_map.get(str(chunk_id))
            if not entry:
                continue
            passage_reference_ids.extend(entry.get("reference_ids", []))
            passage_reference_labels.extend(entry.get("reference_labels", []))
        items.append(
            {
                "passage_id": passage.passage_id,
                "score": passage.score,
                "section_path": list(passage.section_path),
                "chunk_ids": list(passage.chunk_ids),
                "text": passage.text,
                "reference_ids": list(dict.fromkeys(passage_reference_ids)),
                "reference_labels": list(dict.fromkeys(passage_reference_labels)),
            }
        )
    return items


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def link_citations(answer: str, snippets: list[dict[str, object]]) -> str:
    if not answer:
        return answer
    label_map: dict[str, str] = {}
    for snippet in snippets:
        label = snippet.get("citation_label")
        url = snippet.get("download_url")
        if isinstance(label, str) and label.strip() and isinstance(url, str) and url:
            label_map[label.strip()] = url

    if not label_map:
        return answer

    def _replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        url = label_map.get(label)
        if not url:
            return match.group(0)
        return f'<a class="hover:text-indigo-600 underline" href="{url}" target="_blank">[{label}]</a>'

    return _CITATION_RE.sub(_replace, answer)


def link_citations_markdown(answer: str, snippets: list[dict[str, object]]) -> str:
    if not answer:
        return answer
    label_map: dict[str, str] = {}
    for snippet in snippets:
        label = snippet.get("citation_label")
        url = snippet.get("download_url")
        if isinstance(label, str) and label.strip() and isinstance(url, str) and url:
            label_map[label.strip()] = url

    if not label_map:
        return answer

    def _replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        url = label_map.get(label)
        if not url:
            return match.group(0)
        return f"[{label}]({url})"

    return _CITATION_RE.sub(_replace, answer)


def render_markdown_answer(answer: str) -> str:
    if not answer:
        return ""
    try:
        from markdown_it import MarkdownIt

        renderer = MarkdownIt("commonmark", {"html": False, "linkify": True})
        return renderer.render(answer)
    except Exception:
        escaped = html.escape(answer)
        return escaped.replace("\n", "<br>\n")


def _build_download_url(document_id: str | None) -> str | None:
    if not document_id:
        return None
    try:
        from django.urls import reverse

        return reverse("documents:download", args=[document_id])
    except Exception:
        return None


def load_history(state: object) -> list[dict[str, str]]:
    if not isinstance(state, dict):
        return []
    history = state.get("chat_history")
    if not isinstance(history, list):
        return []
    cleaned: list[dict[str, str]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned
