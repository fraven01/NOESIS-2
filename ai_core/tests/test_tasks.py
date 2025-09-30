import uuid
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from ai_core import tasks
from ai_core.infra import object_store
from ai_core.rag import metrics, vector_client
from common import logging as common_logging
from common.celery import ContextTask


def test_split_sentences_prefers_sentence_boundaries() -> None:
    text = "Hallo Welt! Wie geht es dir? Gut."

    result = tasks._split_sentences(text)

    assert result == [
        "Hallo Welt!",
        "Wie geht es dir?",
        "Gut.",
    ]


def test_split_sentences_falls_back_to_paragraphs() -> None:
    text = "Abschnitt eins\n\nAbschnitt zwei\nAbschnitt drei"

    result = tasks._split_sentences(text)

    assert result == ["Abschnitt eins", "Abschnitt zwei", "Abschnitt drei"]


def test_chunkify_applies_overlap_and_limits() -> None:
    sentences = [
        "eins zwei drei",
        "vier fünf sechs",
        "sieben acht neun",
    ]

    chunks = tasks._chunkify(
        sentences,
        target_tokens=6,
        overlap_tokens=2,
        hard_limit=6,
    )

    assert chunks == [
        "eins zwei drei vier fünf sechs",
        "vier fünf sechs sieben acht neun",
        "sieben acht neun",
    ]


def test_chunkify_enforces_hard_limit_and_long_sentences() -> None:
    long_sentence = " ".join(f"wort{i}" for i in range(12))
    sentences = ["kurz eins", long_sentence]

    chunks = tasks._chunkify(
        sentences,
        target_tokens=10,
        overlap_tokens=0,
        hard_limit=4,
    )

    expected_long_chunks = [
        " ".join(f"wort{i}" for i in range(start, min(start + 4, 12)))
        for start in range(0, 12, 4)
    ]

    assert chunks[0] == "kurz eins"
    assert chunks[1:] == expected_long_chunks


def test_build_chunk_prefix_combines_breadcrumbs_and_title() -> None:
    meta = {
        "breadcrumbs": ["Handbuch", " Abschnitt A ", "Teil 1"],
        "title": "Einführung",
    }

    prefix = tasks._build_chunk_prefix(meta)

    assert prefix == "Handbuch / Abschnitt A / Teil 1 — Einführung"


@pytest.mark.usefixtures("rag_database")
def test_upsert_persists_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tenant = str(uuid.uuid4())
    case = str(uuid.uuid4())
    meta = {"tenant": tenant, "case": case, "external_id": "doc-1"}
    vector_client.reset_default_client()

    raw = tasks.ingest_raw(meta, "doc.txt", b"User 123")
    text = tasks.extract_text(meta, raw["path"])
    masked = tasks.pii_mask(meta, text["path"])
    chunks = tasks.chunk(meta, masked["path"])
    embeds = tasks.embed(meta, chunks["path"])

    count = tasks.upsert(meta, embeds["path"])
    assert count == 1

    client = vector_client.get_default_client()
    results = client.search("User", tenant_id=tenant, case_id=case, top_k=5)
    assert len(results) == 1
    assert results[0].content == "User XXX"
    assert results[0].meta.get("hash")
    assert results[0].meta.get("external_id") == "doc-1"
    assert 0.0 <= results[0].meta.get("score", 0.0) <= 1.0

    vector_client.reset_default_client()


@pytest.mark.usefixtures("rag_database")
def test_upsert_no_chunks_is_noop(monkeypatch):
    class _Counter:
        def __init__(self) -> None:
            self.value = 0

        def inc(self, amount: float = 1.0) -> None:
            self.value += amount

    dummy_counter = _Counter()
    monkeypatch.setattr(metrics, "RAG_UPSERT_CHUNKS", dummy_counter)

    written = vector_client.get_default_client().upsert_chunks([])
    assert written == 0
    assert dummy_counter.value == 0
    vector_client.reset_default_client()


def test_task_logging_context_includes_metadata(monkeypatch, tmp_path, settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = False
    monkeypatch.setattr(tasks.object_store, "BASE_PATH", tmp_path)
    original_put_bytes = tasks.object_store.put_bytes

    def _logging_put_bytes(path: str, data: bytes):
        context = common_logging.get_log_context()
        assert context["trace_id"] == "trace-7890"
        assert context["case_id"] == "case-456"
        assert context["tenant"] == "tenant-123"
        assert context["key_alias"] == "alias-1234"
        tasks.logger.info("task-run")
        return original_put_bytes(path, data)

    monkeypatch.setattr(tasks.object_store, "put_bytes", _logging_put_bytes)

    assert isinstance(tasks.ingest_raw._get_current_object(), ContextTask)

    with capture_logs() as logs:
        tasks.ingest_raw(
            {
                "tenant": "tenant-123",
                "case": "case-456",
                "trace_id": "trace-7890",
                "key_alias": "alias-1234",
                "external_id": "doc-logging",
            },
            "doc.txt",
            b"payload",
        )

    events = [entry for entry in logs if entry.get("event") == "task-run"]
    assert events, "expected task-run log entry"
    event = events[0]
    assert event["trace_id"].startswith("tr")
    assert event["trace_id"].endswith("90")
    assert "***" in event["trace_id"]
    assert event["case_id"].startswith("ca")
    assert event["tenant"].startswith("te")
    assert event["key_alias"].startswith("al")
    assert common_logging.get_log_context() == {}


def test_ingest_raw_sanitizes_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = {"tenant": "Tenant Name", "case": "Case*ID", "external_id": "doc-1"}

    result = tasks.ingest_raw(meta, "doc.txt", b"payload")

    safe_tenant = object_store.sanitize_identifier(meta["tenant"])
    safe_case = object_store.sanitize_identifier(meta["case"])
    assert result["path"] == f"{safe_tenant}/{safe_case}/raw/doc.txt"

    stored = tmp_path / ".ai_core_store" / safe_tenant / safe_case / "raw" / "doc.txt"
    assert stored.read_bytes() == b"payload"


def test_ingest_raw_rejects_unsafe_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = {"tenant": "tenant/../", "case": "case", "external_id": "doc-unsafe"}

    original_request = getattr(tasks.ingest_raw, "request", None)
    tasks.ingest_raw.request = SimpleNamespace(headers=None, kwargs=None)

    try:
        with pytest.raises(ValueError):
            tasks.ingest_raw(meta, "doc.txt", b"payload")
    finally:
        if original_request is not None:
            tasks.ingest_raw.request = original_request
        else:
            delattr(tasks.ingest_raw, "request")
