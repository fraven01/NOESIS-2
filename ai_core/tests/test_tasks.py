from ai_core import tasks
from ai_core.rag.vector_client import InMemoryVectorClient


def test_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = {"tenant": "t1", "case": "c1"}

    raw = tasks.ingest_raw(meta, "doc.txt", b"User 123")
    text = tasks.extract_text(meta, raw["path"])
    masked = tasks.pii_mask(meta, text["path"])
    chunks = tasks.chunk(meta, masked["path"])
    embeds = tasks.embed(meta, chunks["path"])

    vc = InMemoryVectorClient()
    monkeypatch.setattr(tasks, "VECTOR_CLIENT", vc)
    count = tasks.upsert(meta, embeds["path"])
    assert count == 1

    masked_file = tmp_path / ".ai_core_store/t1/c1/text/doc.masked.txt"
    assert masked_file.read_text() == "User XXX"

    results = vc.search("User", {"tenant": "t1", "case": "c1"})
    assert len(results) == 1
