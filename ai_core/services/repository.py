"""Repository helpers for document persistence."""

from __future__ import annotations

import logging
from importlib import import_module

from django.conf import settings

from ai_core.adapters.db_documents_repository import DbDocumentsRepository
from documents.repository import DocumentsRepository, InMemoryDocumentsRepository

logger = logging.getLogger(__name__)

_DOCUMENTS_REPOSITORY: DocumentsRepository | None = None


def _build_documents_repository() -> DocumentsRepository:
    """Build the appropriate documents repository."""
    if settings.TESTING:
        return InMemoryDocumentsRepository()

    # Check for explicit repository class configuration
    repository_class_path = getattr(settings, "DOCUMENTS_REPOSITORY_CLASS", None)
    if repository_class_path:
        logger.info(
            "documents_repository_configured",
            extra={"repository_class": repository_class_path},
        )
        try:
            module_path, class_name = repository_class_path.rsplit(".", 1)
            module = import_module(module_path)
            repository_class = getattr(module, class_name)
            return repository_class()
        except Exception:
            logger.exception(
                "documents_repository_instantiation_failed",
                extra={"repository_class": repository_class_path},
            )
            # Don't silently fall back - raise the error
            raise RuntimeError(
                f"Failed to instantiate repository class: {repository_class_path}"
            )

    # Default: Use DB repository for all non-test environments
    logger.info(
        "documents_repository_default",
        extra={"repository_class": "DbDocumentsRepository"},
    )
    return DbDocumentsRepository()


def _get_documents_repository() -> DocumentsRepository:
    try:
        views = import_module("ai_core.views")
        repo = getattr(views, "DOCUMENTS_REPOSITORY", None)
        if isinstance(repo, DocumentsRepository):
            return repo
    except Exception:
        pass

    global _DOCUMENTS_REPOSITORY
    if _DOCUMENTS_REPOSITORY is None:
        _DOCUMENTS_REPOSITORY = _build_documents_repository()
    return _DOCUMENTS_REPOSITORY
