"""Contracts for the crawler ingestion runner view and services."""

from __future__ import annotations

from dataclasses import dataclass, field
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
class CrawlerControlState:
    """Structured control flags for crawler graph execution."""

    snapshot: bool
    snapshot_label: str | None
    fetch: bool
    tags: tuple[str, ...]
    shadow_mode: bool
    dry_run: bool
    mode: str
    review: str | None = None
    manual_review: str | None = None
    force_retire: bool | None = None
    recompute_delta: bool | None = None

    def as_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "snapshot": self.snapshot,
            "snapshot_label": self.snapshot_label,
            "fetch": self.fetch,
            "tags": list(self.tags),
            "shadow_mode": self.shadow_mode,
            "dry_run": self.dry_run,
            "mode": self.mode,
        }
        if self.review:
            payload["review"] = self.review
        if self.manual_review:
            payload["manual_review"] = self.manual_review
        if self.force_retire:
            payload["force_retire"] = True
        if self.recompute_delta:
            payload["recompute_delta"] = True
        return payload


@dataclass(slots=True)
class CrawlerGraphState:
    """Structured crawler graph state assembled for execution."""

    tenant_id: str
    case_id: str | None
    workflow_id: str
    external_id: str
    origin_uri: str
    provider: str
    frontier: dict[str, object]
    fetch: dict[str, object]
    guardrails: dict[str, object]
    document_id: str
    collection_id: str | None
    normalized_document_input: dict[str, object]
    control: CrawlerControlState
    baseline: dict[str, object] = field(default_factory=dict)
    previous_status: str | None = None

    def as_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tenant_id": self.tenant_id,
            "case_id": self.case_id,
            "workflow_id": self.workflow_id,
            "external_id": self.external_id,
            "origin_uri": self.origin_uri,
            "provider": self.provider,
            "frontier": dict(self.frontier),
            "fetch": dict(self.fetch),
            "guardrails": dict(self.guardrails),
            "document_id": self.document_id,
            "collection_id": self.collection_id,
            "normalized_document_input": dict(self.normalized_document_input),
            "control": self.control.as_mapping(),
            "baseline": dict(self.baseline),
        }
        if self.previous_status is not None:
            payload["previous_status"] = self.previous_status
        return payload


@dataclass(slots=True)
class CrawlerStateBundle:
    """Result bundle returned by the crawler state builder service."""

    origin: str
    provider: str
    document_id: str
    state: CrawlerGraphState
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
