import uuid

import pytest

from ai_core import tasks
from ai_core.rag import metrics, vector_client


def test_pipeline_skips_when_rag_disabled(tmp_path, monkeypatch, settings, caplog):
    settings.RAG_ENABLED = False
    monkeypatch.chdir(tmp_path)
    meta = {"tenant": "t1", "case": "c1"}

    raw = tasks.ingest_raw(meta, "doc.txt", b"User 123")
    text = tasks.extract_text(meta, raw["path"])
    masked = tasks.pii_mask(meta, text["path"])
    chunks = tasks.chunk(meta, masked["path"])
    embeds = tasks.embed(meta, chunks["path"])

    with caplog.at_level("INFO"):
        count = tasks.upsert(meta, embeds["path"])
    assert count == 0
    assert "RAG is disabled" in caplog.text

    masked_file = tmp_path / ".ai_core_store/t1/c1/text/doc.masked.txt"
    assert masked_file.read_text() == "User XXX"


@pytest.mark.usefixtures("rag_database")
def test_upsert_persists_chunks_when_rag_enabled(tmp_path, monkeypatch, settings):
    settings.RAG_ENABLED = True
    monkeypatch.chdir(tmp_path)
    tenant = str(uuid.uuid4())
    case = str(uuid.uuid4())
    meta = {"tenant": tenant, "case": case}
    vector_client.reset_default_client()

    raw = tasks.ingest_raw(meta, "doc.txt", b"User 123")
    text = tasks.extract_text(meta, raw["path"])
    masked = tasks.pii_mask(meta, text["path"])
    chunks = tasks.chunk(meta, masked["path"])
    embeds = tasks.embed(meta, chunks["path"])

    count = tasks.upsert(meta, embeds["path"])
    assert count == 1

    client = vector_client.get_default_client()
    results = client.search("User", {"tenant": tenant, "case": case})
    assert len(results) == 1
    assert results[0].content == "User XXX"

    vector_client.reset_default_client()


@pytest.mark.usefixtures("rag_database")
def test_upsert_no_chunks_is_noop(monkeypatch, settings):
    settings.RAG_ENABLED = True

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
