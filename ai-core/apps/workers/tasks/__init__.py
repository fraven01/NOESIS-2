"""Expose task modules for Celery autodiscovery."""

# Import tasks so that Celery registers them when the package is imported.
from . import ingest_raw, extract_text, pii_mask, chunk, embed, upsert  # noqa: F401
