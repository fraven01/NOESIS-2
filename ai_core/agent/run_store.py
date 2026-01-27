from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunStore:
    def save(self, run_id: str, record: dict[str, Any]) -> None:
        raise NotImplementedError

    def get(self, run_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def save(self, run_id: str, record: dict[str, Any]) -> None:
        self._records[run_id] = dict(record)

    def get(self, run_id: str) -> dict[str, Any] | None:
        record = self._records.get(run_id)
        return dict(record) if record is not None else None


class FileRunStore(RunStore):
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, record: dict[str, Any]) -> None:
        path = self._path_for(run_id)
        payload = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        path.write_text(payload, encoding="utf-8")

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self._path_for(run_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _path_for(self, run_id: str) -> Path:
        return self._directory / f"{run_id}.json"


__all__ = ["RunStore", "InMemoryRunStore", "FileRunStore"]
