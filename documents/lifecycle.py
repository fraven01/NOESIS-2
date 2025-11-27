"""Document lifecycle states and transition contracts."""

from __future__ import annotations

from enum import Enum


class DocumentLifecycleState(str, Enum):
    """Simplified lifecycle states for document processing."""

    PENDING = "pending"
    INGESTING = "ingesting"
    EMBEDDED = "embedded"
    ACTIVE = "active"
    FAILED = "failed"
    DELETED = "deleted"


VALID_TRANSITIONS: dict[DocumentLifecycleState, set[DocumentLifecycleState]] = {
    DocumentLifecycleState.PENDING: {
        DocumentLifecycleState.INGESTING,
        DocumentLifecycleState.FAILED,
        DocumentLifecycleState.DELETED,
    },
    DocumentLifecycleState.INGESTING: {
        DocumentLifecycleState.EMBEDDED,
        DocumentLifecycleState.FAILED,
        DocumentLifecycleState.DELETED,
    },
    DocumentLifecycleState.EMBEDDED: {
        DocumentLifecycleState.ACTIVE,
        DocumentLifecycleState.FAILED,
        DocumentLifecycleState.DELETED,
    },
    DocumentLifecycleState.ACTIVE: {
        DocumentLifecycleState.INGESTING,
        DocumentLifecycleState.DELETED,
    },
    DocumentLifecycleState.FAILED: {
        DocumentLifecycleState.PENDING,
        DocumentLifecycleState.DELETED,
    },
    DocumentLifecycleState.DELETED: set(),
}


__all__ = [
    "DocumentLifecycleState",
    "VALID_TRANSITIONS",
]
