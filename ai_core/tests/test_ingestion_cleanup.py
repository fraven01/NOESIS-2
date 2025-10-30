from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from ai_core import ingestion


def _set_object_store_base_path(base_path: Path, monkeypatch: MonkeyPatch) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ingestion.object_store, "BASE_PATH", base_path, raising=False)


def test_cleanup_raw_payload_artifact_removes_normalized_path(tmp_path, monkeypatch):
    base_path = tmp_path / "store"
    _set_object_store_base_path(base_path, monkeypatch)

    target = base_path / "nested" / "payload.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"payload")

    removed = ingestion.cleanup_raw_payload_artifact("nested/./payload.bin")

    assert removed == ["nested/payload.bin"]
    assert not target.exists()


def test_cleanup_raw_payload_artifact_rejects_path_traversal(tmp_path, monkeypatch):
    base_path = tmp_path / "store"
    _set_object_store_base_path(base_path, monkeypatch)

    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    removed = ingestion.cleanup_raw_payload_artifact("../outside.txt")

    assert removed == []
    assert outside.exists()
