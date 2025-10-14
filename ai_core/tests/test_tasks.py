import uuid
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs
from django.conf import settings

from ai_core import tasks
from ai_core.infra import object_store
from ai_core.segmentation import segment_markdown_blocks
from ai_core.rag import metrics, vector_client
from ai_core.rag.ingestion_contracts import IngestionContractErrorCode
from ai_core.tools import InputError
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


def test_segment_markdown_blocks_detects_structures() -> None:
    document = """# Titel\n\nEin Absatz mit Text.\n\n- Punkt eins\n- Punkt zwei\n\n```python\nprint('x')\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"""

    segments = segment_markdown_blocks(document)

    assert segments[:5] == [
        "# Titel",
        "Ein Absatz mit Text.",
        "- Punkt eins\n- Punkt zwei",
        "```python\nprint('x')\n```",
        "| A | B |\n| --- | --- |\n| 1 | 2 |",
    ]


def test_chunkify_applies_overlap_and_limits() -> None:
    sentences = [
        "eins zwei drei",
        "vier fünf sechs",
        "sieben acht neun",
    ]

    with tasks.force_whitespace_tokenizer():
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

    with tasks.force_whitespace_tokenizer():
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


def test_resolve_overlap_respects_zero_configuration(monkeypatch) -> None:
    meta = {"tenant_id": "tenant", "case_id": "case"}
    text = "Dies ist ein narrativer Text, der normalerweise eine Überlappung hätte."
    target_tokens = 200
    hard_limit = 256

    monkeypatch.setattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", 0, raising=False)

    with tasks.force_whitespace_tokenizer():
        overlap = tasks._resolve_overlap_tokens(
            text,
            meta,
            target_tokens=target_tokens,
            hard_limit=hard_limit,
        )

    assert overlap == 0


def test_chunk_dynamic_overlap_varies_by_document_style(monkeypatch) -> None:
    narrative_meta = {
        "tenant_id": "tenant-dynamic",
        "case_id": "case-dynamic",
        "external_id": "story-1",
        "doc_class": "narrative",
    }
    faq_meta = {**narrative_meta, "external_id": "faq-1", "doc_class": "faq"}

    narrative_text = (
        "Ich ging gestern durch den Park und erinnerte mich daran, "
        "wie wir gemeinsam die langen Wege entlang spaziert sind. "
        "Wir sprachen darüber, was uns bewegt und wohin wir als nächstes gehen."
    )
    faq_text = (
        "Frage: Wie bestelle ich?\nAntwort: Wählen Sie einen Artikel aus.\n"
        "- Schritt 1: Produkt wählen\n- Schritt 2: In den Warenkorb legen\n"
        "- Schritt 3: Bestellung abschließen"
    )

    narrative_path = tasks._build_path(narrative_meta, "text", "story.txt")
    faq_path = tasks._build_path(faq_meta, "text", "faq.txt")
    object_store.put_bytes(narrative_path, narrative_text.encode("utf-8"))
    object_store.put_bytes(faq_path, faq_text.encode("utf-8"))

    original_chunkify = tasks._chunkify

    def _capture(overlaps: list[int]):
        def _wrapper(
            sentences,
            *,
            target_tokens,
            overlap_tokens,
            hard_limit,
        ):
            overlaps.append(overlap_tokens)
            return original_chunkify(
                sentences,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                hard_limit=hard_limit,
            )

        return _wrapper

    narrative_overlaps: list[int] = []
    faq_overlaps: list[int] = []

    with tasks.force_whitespace_tokenizer():
        monkeypatch.setattr(tasks, "_chunkify", _capture(narrative_overlaps))
        tasks.chunk(narrative_meta, narrative_path)

        monkeypatch.setattr(tasks, "_chunkify", _capture(faq_overlaps))
        tasks.chunk(faq_meta, faq_path)

        monkeypatch.setattr(tasks, "_chunkify", original_chunkify)

    assert narrative_overlaps, "expected overlap capture for narrative document"
    assert faq_overlaps, "expected overlap capture for FAQ document"

    target_tokens = int(getattr(settings, "RAG_CHUNK_TARGET_TOKENS", 450))
    hard_limit = max(target_tokens, 512)
    narrative_expected = tasks._resolve_overlap_tokens(
        narrative_text,
        narrative_meta,
        target_tokens=target_tokens,
        hard_limit=hard_limit,
    )
    faq_expected = tasks._resolve_overlap_tokens(
        faq_text,
        faq_meta,
        target_tokens=target_tokens,
        hard_limit=hard_limit,
    )

    assert max(narrative_overlaps) == narrative_expected
    assert max(faq_overlaps) == faq_expected
    assert narrative_expected > faq_expected

    for overlap in (narrative_expected, faq_expected):
        ratio = overlap / max(1, target_tokens)
        assert 0.10 <= ratio <= 0.25

    stored_root = object_store.BASE_PATH / narrative_meta["tenant_id"]
    if stored_root.exists():
        import shutil

        shutil.rmtree(stored_root)


def test_chunk_uses_structured_blocks_and_limit() -> None:
    meta = {"tenant_id": "tenant", "case_id": "case", "external_id": "doc"}
    document = (
        "# Titel\n\nEin Absatz.\n\n- Punkt eins\n- Punkt zwei\n\n"
        "```python\nprint('x')\n```\n\n"
        + " ".join(f"wort{i}" for i in range(600))
    )
    text_path = tasks._build_path(meta, "text", "doc.txt")
    object_store.put_bytes(text_path, document.encode("utf-8"))

    with tasks.force_whitespace_tokenizer():
        result = tasks.chunk(meta, text_path)

    chunk_records = object_store.read_json(result["path"])
    contents = [entry["content"] for entry in chunk_records]

    assert contents[0].strip() == "# Titel"
    assert any("```python" in content for content in contents)
    assert any("Punkt eins" in content for content in contents)

    long_chunks = [content for content in contents if "wort" in content]
    assert len(long_chunks) > 1
    for chunk_text in long_chunks:
        assert len([token for token in chunk_text.split() if token]) <= 512

    stored_root = object_store.BASE_PATH / meta["tenant_id"]
    if stored_root.exists():
        import shutil

        shutil.rmtree(stored_root)


def test_build_chunk_prefix_combines_breadcrumbs_and_title() -> None:
    meta = {
        "breadcrumbs": ["Handbuch", " Abschnitt A ", "Teil 1"],
        "title": "Einführung",
    }

    prefix = tasks._build_chunk_prefix(meta)

    assert prefix == "Handbuch / Abschnitt A / Teil 1 — Einführung"


@pytest.mark.skipif(
    str(getattr(settings, "PII_MODE", "industrial")).lower() == "off"
    or str(getattr(settings, "PII_POLICY", "balanced")).lower() == "off",
    reason="PII masking disabled in settings",
)
@pytest.mark.usefixtures("rag_database")
def test_upsert_persists_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tenant = str(uuid.uuid4())
    case = str(uuid.uuid4())
    meta = {"tenant_id": tenant, "case_id": case, "external_id": "doc-1"}
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


def test_upsert_forwards_tenant_schema(monkeypatch):
    meta = {
        "tenant_id": "tenant-42",
        "tenant_schema": "schema-tenant-42",
    }

    embeddings = [
        {
            "content": "payload",
            "embedding": [0.0],
            "meta": {"tenant_id": "tenant-42"},
        }
    ]

    monkeypatch.setattr(tasks.object_store, "read_json", lambda path: embeddings)

    class _Router:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def for_tenant(self, tenant_id: str, tenant_schema: str | None = None):
            self.calls.append((tenant_id, tenant_schema))
            return self

        def upsert_chunks(self, chunks):
            return len(list(chunks))

    router = _Router()
    monkeypatch.setattr(tasks, "get_default_router", lambda: router)

    written = tasks.upsert(meta, "embeddings.json")

    assert written == 1
    assert router.calls == [("tenant-42", "schema-tenant-42")]


def test_upsert_raises_on_dimension_mismatch(monkeypatch):
    meta = {
        "tenant_id": "tenant-42",
        "embedding_profile": "standard",
        "vector_space_id": "global",
        "vector_space_dimension": 2,
        "process": "review",
        "doc_class": "manual",
    }
    embeddings = [
        {
            "content": "payload",
            "embedding": [0.0],
            "meta": {
                "tenant_id": "tenant-42",
                "embedding_profile": "standard",
                "vector_space_id": "global",
                "external_id": "doc-1",
            },
        }
    ]

    monkeypatch.setattr(tasks.object_store, "read_json", lambda path: embeddings)

    class _Router:
        def upsert_chunks(self, chunks):  # pragma: no cover - should not be called
            raise AssertionError("upsert_chunks should not be invoked on mismatch")

    monkeypatch.setattr(tasks, "get_default_router", lambda: _Router())

    with pytest.raises(InputError) as excinfo:
        tasks.upsert(meta, "embeddings.json")

    error = excinfo.value
    assert error.code == IngestionContractErrorCode.VECTOR_DIMENSION_MISMATCH
    assert error.context["process"] == "review"
    assert error.context["doc_class"] == "manual"
    assert error.context["embedding_profile"] == "standard"
    assert error.context["vector_space_id"] == "global"
    assert error.context["observed_dimension"] == 1


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
                "tenant_id": "tenant-123",
                "case_id": "case-456",
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
    meta = {
        "tenant_id": "Tenant Name",
        "case_id": "Case*ID",
        "external_id": "doc-1",
    }

    result = tasks.ingest_raw(meta, "doc.txt", b"payload")

    safe_tenant = object_store.sanitize_identifier(meta["tenant_id"])
    safe_case = object_store.sanitize_identifier(meta["case_id"])
    assert result["path"] == f"{safe_tenant}/{safe_case}/raw/doc.txt"

    stored = tmp_path / ".ai_core_store" / safe_tenant / safe_case / "raw" / "doc.txt"
    assert stored.read_bytes() == b"payload"


def test_ingest_raw_rejects_unsafe_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = {
        "tenant_id": "tenant/../",
        "case_id": "case",
        "external_id": "doc-unsafe",
    }

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
