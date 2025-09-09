import importlib
import os
import sys


def _import_celery():
    """Reload the Celery module to apply environment changes."""
    sys.modules.pop("noesis2.celery", None)
    return importlib.import_module("noesis2.celery")


def test_uses_development_settings_by_default(monkeypatch):
    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    _import_celery()
    assert os.environ["DJANGO_SETTINGS_MODULE"] == "noesis2.settings.development"


def test_respects_explicit_env_setting(monkeypatch):
    monkeypatch.setenv("DJANGO_SETTINGS_MODULE", "noesis2.settings.production")
    _import_celery()
    assert os.environ["DJANGO_SETTINGS_MODULE"] == "noesis2.settings.production"
