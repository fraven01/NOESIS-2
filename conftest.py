from pathlib import Path

import pytest
from django.conf import settings

try:
    engine = settings.DATABASES["default"]["ENGINE"]
except Exception:  # pragma: no cover - settings not configured
    engine = None

if engine != "django_tenants.postgresql_backend":
    pytest.skip("requires django_tenants.postgresql_backend", allow_module_level=True)


@pytest.fixture(autouse=True)
def tmp_media_root(tmp_path, settings):
    """Store uploaded files under a per-test temporary MEDIA_ROOT."""
    media = tmp_path / "media"
    media.mkdir(parents=True, exist_ok=True)
    settings.MEDIA_ROOT = str(media)
    yield


@pytest.fixture(autouse=True)
def disable_auto_create_schema(monkeypatch):
    monkeypatch.setattr("customers.models.Tenant.auto_create_schema", False)


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
