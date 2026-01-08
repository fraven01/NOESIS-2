from __future__ import annotations

from pathlib import Path


def test_persistence_paths_do_not_cast_user_id_to_int() -> None:
    paths = [
        Path("documents/domain_service.py"),
        Path("ai_core/adapters/db_documents_repository.py"),
    ]
    disallowed = ("int(created_by_user_id)", "int(user_id)")

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in disallowed:
            assert token not in text
