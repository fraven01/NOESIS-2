from __future__ import annotations

from types import SimpleNamespace

from ai_core.tasks.ingestion_tasks import _cap_header_text, _resolve_context_header


def _stub_context():
    scope = SimpleNamespace(tenant_id="tenant-1", trace_id="trace-1")
    business = SimpleNamespace(case_id=None)
    return SimpleNamespace(scope=scope, business=business)


def test_resolve_context_header_heuristic_combines_title_and_section() -> None:
    context = _stub_context()
    cache: dict[str, str] = {}

    header = _resolve_context_header(
        mode="heuristic",
        title="Handbook",
        section_label="Chapter 1",
        chunk_preview="preview",
        context=context,
        cache=cache,
    )

    assert header == "Handbook | Chapter 1"
    assert cache


def test_cap_header_text_respects_limits() -> None:
    header = _cap_header_text(
        "One two three four five six seven eight nine ten",
        max_chars=20,
        max_words=4,
    )

    assert header == "One two three four"
