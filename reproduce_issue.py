import re
from typing import Mapping


def _resolve_citation_label(snippet: Mapping[str, object]) -> str:
    raw_label = snippet.get("citation")
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()
    for key in ("source", "id"):
        candidate = snippet.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "Source"


def _build_download_url(document_id):
    return f"/download/{document_id}"


def build_snippet_items(snippets, limit=None):
    items = []
    if limit is None or limit <= 0:
        limit = 3
    for snippet in snippets[:limit]:
        document_id = snippet.get("document_id")
        download_url = _build_download_url(document_id)
        source = snippet.get("source") or "Unknown"
        text = snippet.get("text") or ""
        items.append(
            {
                "source": source,
                "title": str(text)[:200],
                "score_percent": 0,
                "download_url": download_url,
                "citation_label": _resolve_citation_label(snippet),
            }
        )
    return items


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def link_citations(answer: str, snippets: list[dict[str, object]]) -> str:
    if not answer:
        return answer
    label_map = {}
    print("DEBUG: Snippets:", snippets)
    for snippet in snippets:
        label = snippet.get("citation_label")
        url = snippet.get("download_url")
        if isinstance(label, str) and label.strip() and isinstance(url, str) and url:
            label_map[label.strip()] = url

    print("DEBUG: Label Map:", label_map)

    if not label_map:
        return answer

    def _replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        url = label_map.get(label)
        if not url:
            return match.group(0)
        return f'<a class="hover:text-indigo-600 underline" href="{url}" target="_blank">[{label}]</a>'

    return _CITATION_RE.sub(_replace, answer)


# Test Data
used_sources = [
    {"id": "s1", "label": "Doc 1", "relevance_score": 0.9},
    {"id": "s2", "label": "Doc 2", "relevance_score": 0.8},
    {"id": "s3", "label": "Doc 3", "relevance_score": 0.7},
    {"id": "s4", "label": "Doc 4", "relevance_score": 0.6},
]
snippets = [
    {"document_id": "d1", "text": "t1", "source": "Doc 1", "score": 0.9},
    {"document_id": "d2", "text": "t2", "source": "Doc 2", "score": 0.8},
    {"document_id": "d3", "text": "t3", "source": "Doc 3", "score": 0.7},
    {"document_id": "d4", "text": "t4", "source": "Doc 4", "score": 0.6},
]
answer = "Answer [Doc 1]"
top_k_effective = 3

snippet_limit = top_k_effective or len(snippets) or None
snippet_items = build_snippet_items(snippets, limit=snippet_limit)
final_answer = link_citations(answer, snippet_items)

print(f"Final Answer: {final_answer}")
assert "/download/d1" in final_answer
