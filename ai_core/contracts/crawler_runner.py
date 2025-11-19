"""Contracts for the crawler ingestion runner view and services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ai_core.schemas import CrawlerRunRequest


@dataclass(frozen=True)
class CrawlerRunContext:
    """Request-scoped metadata used while composing crawler state."""

    meta: Mapping[str, object]
    request: CrawlerRunRequest
    workflow_id: str
    repository: object | None = None


@dataclass(slots=True)
class CrawlerStateBundle:
    """Result bundle returned by the crawler state builder service."""

    origin: str
    provider: str
    document_id: str
    state: dict[str, object]
    fetch_used: bool
    http_status: int | None
    fetched_bytes: int | None
    media_type_effective: str | None
    fetch_elapsed: float | None
    fetch_retries: int | None
    fetch_retry_reason: str | None
    fetch_backoff_total_ms: float | None
    snapshot_path: str | None
    snapshot_sha256: str | None
    tags: tuple[str, ...]
    collection_id: str | None
    snapshot_requested: bool
    snapshot_label: str | None
    review: str | None
    dry_run: bool


class CrawlerRunError(RuntimeError):
    """Raised when crawler state preparation cannot proceed."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})
