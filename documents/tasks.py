"""Celery tasks for document upload processing."""

from celery import shared_task
from typing import Any, Dict

from documents.upload_worker import UploadWorker


@shared_task(queue="ingestion")
def upload_document_task(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    metadata: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Process uploaded document asynchronously.

    Args:
        file_bytes: Raw file content
        filename: Name of the file
        content_type: MIME type
        metadata: Document metadata
        meta: Request context metadata
    """
    # Mock UploadedFile for Worker compatibility
    from io import BytesIO

    class MockUploadedFile:
        def __init__(self, data: bytes, name: str, content_type: str):
            self.file = BytesIO(data)
            self.name = name
            self.content_type = content_type
            self.size = len(data)

        def read(self) -> bytes:
            self.file.seek(0)
            return self.file.read()

    upload = MockUploadedFile(file_bytes, filename, content_type)

    worker = UploadWorker()
    scope_context = meta["scope_context"]
    result = worker.process(
        upload,
        tenant_id=scope_context["tenant_id"],
        case_id=scope_context.get("case_id"),
        trace_id=scope_context.get("trace_id"),
        invocation_id=scope_context.get("invocation_id"),
        document_metadata=metadata,
        ingestion_overrides=metadata.get("ingestion_overrides"),
    )

    return {
        "status": result.status,
        "document_id": result.document_id,
        "task_id": result.task_id,
    }
