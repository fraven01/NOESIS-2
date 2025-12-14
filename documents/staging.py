"""File staging infrastructure for document ingestion.

This module manages the download of remote document blobs to local temporary storage
to support efficient processing by file-based parsers.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from documents.contracts import (
    ExternalBlob,
    FileBlob,
    LocalFileBlob,
    NormalizedDocument,
)

logger = logging.getLogger(__name__)

# Base directory for ingestion staging
# Can be overridden by env var for container setups
STAGING_DIR = os.environ.get("INGESTION_STAGING_DIR", tempfile.gettempdir())


class FileStager:
    """Manages temporary local files for ingestion processing."""

    def __init__(self, base_dir: Optional[str] = None):
        self._base_dir = Path(base_dir or STAGING_DIR) / "ingestion_staging"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def stage(self, document: NormalizedDocument, storage: Any) -> NormalizedDocument:
        """Download document blob to a local temporary file if necessary.

        If the document is already local or inline, returns it as-is.
        If it's a FileBlob or ExternalBlob, downloads content and returns a new
        NormalizedDocument pointing to the local file.
        """
        if not isinstance(document.blob, (FileBlob, ExternalBlob)):
            return document

        try:
            # Generate safe temp filename
            blob_ext = self._guess_extension(document)
            temp_filename = f"{uuid.uuid4()}{blob_ext}"
            temp_path = self._base_dir / temp_filename

            logger.info(
                "staging_file_start",
                extra={
                    "document_id": str(document.ref.document_id),
                    "blob_type": document.blob.type,
                    "target_path": str(temp_path),
                },
            )

            # Resolve payload bytes (downloads from storage)
            # TODO: In future, use stream_to_file interface if available on storage
            from documents.normalization import document_payload_bytes

            payload = document_payload_bytes(document, storage=storage)

            # Write to disk
            with open(temp_path, "wb") as f:
                f.write(payload)

            # Create LocalFileBlob
            local_blob = LocalFileBlob(
                type="local_file",
                path=str(temp_path.absolute()),
                media_type=getattr(document.blob, "media_type", None)
                or (document.meta.pipeline_config or {}).get("media_type"),
            )

            # Return updated document
            return document.model_copy(update={"blob": local_blob})

        except Exception as exc:
            logger.exception(
                "staging_file_failed",
                extra={"document_id": str(document.ref.document_id), "error": str(exc)},
            )
            raise

    def cleanup(self, document: NormalizedDocument) -> None:
        """Remove the temporary local file associated with the document."""
        if isinstance(document.blob, LocalFileBlob):
            try:
                path = Path(document.blob.path)
                # Security check: ensure we are only deleting files within our staging dir
                # to prevent arbitrary file deletion
                if self._base_dir.resolve() in path.resolve().parents:
                    if path.exists():
                        path.unlink()
                        logger.info(
                            "staging_cleanup_success",
                            extra={"path": str(path)},
                        )
                else:
                    logger.warning(
                        "staging_cleanup_skipped_unsafe",
                        extra={"path": str(path), "staging_root": str(self._base_dir)},
                    )
            except Exception as exc:
                logger.error(
                    "staging_cleanup_failed",
                    extra={"path": document.blob.path, "error": str(exc)},
                )

    def _guess_extension(self, document: NormalizedDocument) -> str:
        """Infer file extension from metadata or media type."""
        # Try from source filename/URI
        if isinstance(document.blob, FileBlob):
            path = Path(document.blob.uri)
            if path.suffix:
                return path.suffix

        # Try from media type
        media_type = None

        if document.meta and document.meta.external_ref:
            media_type = document.meta.external_ref.get("media_type")

        if not media_type:
            media_type = getattr(document.blob, "media_type", None) or (
                document.meta.pipeline_config or {}
            ).get("media_type")

        if media_type == "text/html":
            return ".html"
        if media_type == "application/pdf":
            return ".pdf"

        return ".bin"
