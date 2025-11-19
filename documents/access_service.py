"""Document access service for tenant-isolated file retrieval.

Business logic layer that handles authorization and file resolution,
separated from HTTP protocol concerns.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from uuid import UUID

from structlog.stdlib import get_logger

from .utils import get_upload_file_path

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentAccessResult:
    """Result of document access check and file resolution."""

    document: Any  # Repository document object
    blob_path: Path
    file_size: int
    mtime: float

    @property
    def document_id(self) -> UUID:
        """Extract document ID from repository object."""
        return self.document.ref.document_id

    @property
    def tenant_id(self) -> str:
        """Extract tenant ID from repository object."""
        return self.document.ref.tenant_id

    @property
    def workflow_id(self) -> str:
        """Extract workflow ID from repository object."""
        return self.document.ref.workflow_id


@dataclass(frozen=True)
class AccessError:
    """Structured error for access failures."""

    status_code: int
    error_code: str
    message: str


class DocumentAccessService:
    """Service for tenant-isolated document access and file resolution.

    Handles:
    - Document lookup via repository
    - Tenant isolation validation
    - Physical file path resolution
    - File existence checks
    """

    def __init__(self, repository: Any):
        """Initialize with document repository.

        Args:
            repository: Documents repository (duck-typed for testability)
        """
        self._repository = repository

    def get_document_for_download(
        self,
        tenant_id: str,
        document_id: UUID,
    ) -> tuple[Optional[DocumentAccessResult], Optional[AccessError]]:
        """Retrieve document with tenant isolation and file validation.

        Args:
            tenant_id: Requesting tenant ID
            document_id: Document UUID to retrieve

        Returns:
            Tuple of (result, error). One is always None.
            - Success: (DocumentAccessResult, None)
            - Failure: (None, AccessError)

        Business rules:
        1. Document must exist in repository
        2. Document's tenant must match requesting tenant
        3. Physical file must exist on disk
        """
        # 1. Repository lookup
        doc = self._repository.get(tenant_id, document_id)

        if not doc:
            logger.warning(
                "document.access.not_found",
                tenant_id=tenant_id,
                document_id=str(document_id),
            )
            return None, AccessError(
                status_code=404,
                error_code="DocumentNotFound",
                message=f"Document {document_id} not found",
            )

        # 2. Tenant isolation check
        if doc.ref.tenant_id != tenant_id:
            logger.error(
                "document.access.tenant_mismatch",
                tenant_id=tenant_id,
                document_tenant_id=doc.ref.tenant_id,
                document_id=str(document_id),
            )
            return None, AccessError(
                status_code=403,
                error_code="TenantMismatch",
                message="Access denied",
            )

        # 3. Physical file resolution
        blob_path = get_upload_file_path(
            doc.ref.tenant_id,
            doc.ref.workflow_id,
            str(doc.ref.document_id),
        )

        if not blob_path.exists():
            logger.error(
                "document.access.blob_missing",
                tenant_id=tenant_id,
                document_id=str(document_id),
                blob_path=str(blob_path),
            )
            return None, AccessError(
                status_code=404,
                error_code="BlobNotFound",
                message="Document file not found on disk",
            )

        # 4. File stats
        st = os.stat(blob_path)

        result = DocumentAccessResult(
            document=doc,
            blob_path=blob_path,
            file_size=st.st_size,
            mtime=st.st_mtime,
        )

        logger.info(
            "document.access.success",
            tenant_id=tenant_id,
            document_id=str(document_id),
            file_size=st.st_size,
        )

        return result, None
