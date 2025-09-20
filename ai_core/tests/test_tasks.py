import io
import logging
import uuid

import pytest

from ai_core import tasks
from ai_core.rag import metrics, vector_client
from common import logging as common_logging


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


def test_task_logging_context_includes_metadata(monkeypatch, tmp_path, settings):
    monkeypatch.setattr(tasks.object_store, "BASE_PATH", tmp_path)
    original_put_bytes = tasks.object_store.put_bytes

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.addFilter(common_logging.RequestTaskContextFilter())
    formatter = logging.Formatter(settings.LOGGING["formatters"]["verbose"]["format"])
    handler.setFormatter(formatter)

    logger = logging.getLogger("ai_core.tests.tasks")
    original_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    def _logging_put_bytes(path: str, data: bytes):
        logger.info("task-run")
        return original_put_bytes(path, data)

    monkeypatch.setattr(tasks.object_store, "put_bytes", _logging_put_bytes)

    try:
        tasks.ingest_raw(
            {
                "tenant": "tenant-123",
                "case": "case-456",
                "trace_id": "trace-7890",
                "key_alias": "alias-1234",
            },
            "doc.txt",
            b"payload",
        )
    finally:
        logger.removeHandler(handler)
        logger.propagate = original_propagate

    output = log_buffer.getvalue()
    assert "task-run" in output
    assert "trace=-" not in output
    assert "case=-" not in output
    assert "tenant=-" not in output
    assert "key_alias=-" not in output
