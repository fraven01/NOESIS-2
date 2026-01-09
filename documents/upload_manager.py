"""Manager for handling document upload orchestration.

This module provides the high-level interface for handling upload requests,
standardizing the input, and dispatching to the asynchronous worker.
"""

from __future__ import annotations

from typing import Any, Dict

from ai_core.tool_contracts.base import tool_context_from_meta


class UploadManager:
    """Technical Manager for Upload operations (Layer 3)."""

    def dispatch_upload_request(
        self,
        upload: Any,  # Duck-typed UploadedFile
        metadata: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch upload to async worker.

        Returns immediately with task_id (HTTP 202).

        Args:
            upload: File-like object
            metadata: Document metadata
            meta: Request context metadata (tenant, trace, etc.)

        Returns:
            Dict containing status, task_id, and document_id
        """
        from documents.tasks import upload_document_task

        # Serialize upload file for Celery
        # Note: For very large files, this memory-heavy approach might need
        # revisiting (e.g. streaming to S3 presigned URL first), but for
        # standard API uploads < 100MB this is standard practice.
        file_bytes = upload.read()

        # Extract context for tracing and task_id
        from uuid import uuid4
        from common.celery import with_scope_apply_async

        context = tool_context_from_meta(meta)
        ingestion_run_id = context.scope.ingestion_run_id or uuid4().hex

        # Dispatch to Celery with scope propagation
        signature = upload_document_task.s(
            file_bytes=file_bytes,
            filename=upload.name,
            content_type=getattr(upload, "content_type", "application/octet-stream"),
            metadata=metadata,
            meta=meta,
        )
        task_result = with_scope_apply_async(
            signature,
            context.scope.model_dump(mode="json", exclude_none=True),
            task_id=ingestion_run_id,
        )

        return {
            "status": "accepted",
            "task_id": task_result.id,
            "trace_id": context.scope.trace_id,
            "document_id": metadata.get(
                "document_id"
            ),  # Might be None if not pre-provided
            "message": "Upload accepted for processing",
        }
