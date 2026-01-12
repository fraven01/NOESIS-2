from __future__ import annotations

import os
from collections.abc import Mapping


DEFAULT_HISTORY_LIMIT = 6


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


def coerce_optional_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "on"}
    return False


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


def build_snippet_items(snippets: list[dict]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for snippet in snippets[:3]:
        document_id = snippet.get("document_id")
        if not isinstance(document_id, str):
            meta = snippet.get("meta")
            if isinstance(meta, dict):
                document_id = meta.get("document_id")
        download_url = _build_download_url(
            document_id if isinstance(document_id, str) else None
        )
        source = snippet.get("source") or "Unknown"
        text = snippet.get("text") or ""
        try:
            score_value = float(snippet.get("score", 0))
        except (TypeError, ValueError):
            score_value = 0.0
        items.append(
            {
                "source": source,
                "title": str(text)[:200],
                "score_percent": int(score_value * 100),
                "download_url": download_url,
            }
        )
    return items


def _build_download_url(document_id: str | None) -> str | None:
    if not document_id:
        return None
    try:
        from django.urls import reverse

        return reverse("documents:download", args=[document_id])
    except Exception:
        return None


def resolve_history_limit() -> int:
    value = coerce_optional_int(os.getenv("RAG_CHAT_HISTORY_MAX_MESSAGES"), minimum=1)
    return value or DEFAULT_HISTORY_LIMIT


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


def append_history(
    history: list[dict[str, str]],
    *,
    role: str,
    content: str | None,
) -> None:
    if not content:
        return
    history.append({"role": role, "content": content})


def trim_history(history: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return history
    if len(history) <= limit:
        return history
    return history[-limit:]
