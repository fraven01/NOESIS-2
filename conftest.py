from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def tmp_media_root(tmp_path, settings):
    """Store uploaded files under a per-test temporary MEDIA_ROOT."""
    media = tmp_path / "media"
    media.mkdir(parents=True, exist_ok=True)
    settings.MEDIA_ROOT = str(media)
    yield


@pytest.fixture(autouse=True, scope="session")
def cleanup_documents_test_files_session():
    """Safety net: remove stray documents/test*.txt in repo root, pre/post session."""
    repo_docs = Path(__file__).resolve().parent / "documents"
    for phase in ("pre", "post"):
        for p in repo_docs.glob("test*.txt"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        if phase == "pre":
            yield
