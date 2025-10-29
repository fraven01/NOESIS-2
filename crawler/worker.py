"""Crawler worker implementation forwarding fetch results to the AI Core."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from ai_core.tasks import run_ingestion_graph

from .errors import CrawlerError
from .fetcher import FetchRequest, FetchResult, FetchStatus

IngestionTask = Callable[[Mapping[str, Any], Optional[Mapping[str, Any]]], Any]


@dataclass(frozen=True)
class WorkerPublishResult:
    """Result payload returned after attempting to publish a fetch outcome."""

    status: str
    fetch_result: FetchResult
    task_id: Optional[str] = None
    error: Optional[CrawlerError] = None

    @property
    def published(self) -> bool:
        """Return ``True`` when the ingestion task was enqueued."""

        return self.status == "published"


class CrawlerWorker:
    """Thin worker delegating post-fetch processing to the AI Core graph."""

    def __init__(
        self,
        fetcher: Any,
        *,
        ingestion_task: Optional[IngestionTask] = None,
    ) -> None:
        if ingestion_task is None:
            ingestion_task = run_ingestion_graph
        self._fetcher = fetcher
        self._ingestion_task = ingestion_task

    def process(
        self,
        request: FetchRequest,
        *,
        tenant_id: str,
        case_id: Optional[str] = None,
        crawl_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        request_id: Optional[str] = None,
        frontier_state: Optional[Mapping[str, Any]] = None,
        document_id: Optional[str] = None,
        document_metadata: Optional[Mapping[str, Any]] = None,
        ingestion_overrides: Optional[Mapping[str, Any]] = None,
        meta_overrides: Optional[Mapping[str, Any]] = None,
    ) -> WorkerPublishResult:
        """Fetch ``request`` and publish the payload to the ingestion graph."""

        result = self._fetcher.fetch(request)
        if result.status is not FetchStatus.FETCHED:
            return WorkerPublishResult(
                status=result.status.value,
                fetch_result=result,
                error=result.error,
            )

        payload_state = self._compose_state(
            result,
            request,
            tenant_id=tenant_id,
            case_id=case_id,
            request_id=request_id,
            frontier_state=frontier_state,
            document_id=document_id,
            document_metadata=document_metadata,
            ingestion_overrides=ingestion_overrides,
            crawl_id=crawl_id,
        )
        meta_payload = self._compose_meta(
            tenant_id=tenant_id,
            case_id=case_id,
            crawl_id=crawl_id,
            idempotency_key=idempotency_key,
            request_id=request_id,
            frontier_state=frontier_state,
            meta_overrides=meta_overrides,
        )

        async_result = self._ingestion_task.delay(payload_state, meta_payload)
        task_id = getattr(async_result, "id", None)
        return WorkerPublishResult(
            status="published",
            fetch_result=result,
            task_id=task_id,
        )

    def _compose_state(
        self,
        result: FetchResult,
        request: FetchRequest,
        *,
        tenant_id: str,
        case_id: Optional[str],
        request_id: Optional[str],
        frontier_state: Optional[Mapping[str, Any]],
        document_id: Optional[str],
        document_metadata: Optional[Mapping[str, Any]],
        ingestion_overrides: Optional[Mapping[str, Any]],
        crawl_id: Optional[str],
    ) -> dict[str, Any]:
        state: dict[str, Any] = dict(ingestion_overrides or {})
        state.setdefault("tenant_id", tenant_id)
        if case_id is not None:
            state.setdefault("case_id", case_id)
        if request_id is not None:
            state.setdefault("request_id", request_id)
        if crawl_id is not None:
            state.setdefault("crawl_id", crawl_id)
        state.setdefault("frontier", dict(frontier_state or {}))

        raw_meta: dict[str, Any] = dict(document_metadata or {})
        raw_meta.setdefault("origin_uri", request.canonical_source)
        raw_meta.setdefault("provider", raw_meta.get("provider", "web"))
        if result.metadata.content_type and "content_type" not in raw_meta:
            raw_meta["content_type"] = result.metadata.content_type
        if result.metadata.status_code is not None and "status_code" not in raw_meta:
            raw_meta["status_code"] = result.metadata.status_code
        if result.metadata.etag and "etag" not in raw_meta:
            raw_meta["etag"] = result.metadata.etag
        if result.metadata.last_modified and "last_modified" not in raw_meta:
            raw_meta["last_modified"] = result.metadata.last_modified
        if result.metadata.content_length is not None and "content_length" not in raw_meta:
            raw_meta["content_length"] = result.metadata.content_length

        payload_bytes = bytes(result.payload or b"")
        payload_base64 = base64.b64encode(payload_bytes).decode("ascii")

        raw_document: dict[str, Any] = {
            "metadata": raw_meta,
            "payload_base64": payload_base64,
        }
        if document_id is not None:
            raw_document["document_id"] = document_id
        elif "document_id" in raw_meta:
            raw_document["document_id"] = raw_meta["document_id"]

        state["raw_document"] = raw_document
        state.setdefault("fetch", self._summarize_fetch(result))
        return state

    def _compose_meta(
        self,
        *,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        idempotency_key: Optional[str],
        request_id: Optional[str],
        frontier_state: Optional[Mapping[str, Any]],
        meta_overrides: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tenant_id": tenant_id,
            "case_id": case_id,
            "crawl_id": crawl_id,
            "idempotency_key": idempotency_key,
            "request_id": request_id,
        }
        if frontier_state:
            payload.setdefault("frontier", dict(frontier_state))
        if meta_overrides:
            payload.update(dict(meta_overrides))
        return {key: value for key, value in payload.items() if value is not None}

    def _summarize_fetch(self, result: FetchResult) -> dict[str, Any]:
        telemetry = result.telemetry
        summary: dict[str, Any] = {
            "status": result.status.value,
            "status_code": result.metadata.status_code,
            "content_type": result.metadata.content_type,
            "latency": telemetry.latency,
            "bytes_downloaded": telemetry.bytes_downloaded,
            "retries": telemetry.retries,
            "retry_reason": telemetry.retry_reason,
            "backoff_total_ms": telemetry.backoff_total_ms,
            "policy_events": list(result.policy_events),
        }
        if result.detail:
            summary["detail"] = result.detail
        if result.error is not None:
            summary["error"] = self._serialize_error(result.error)
        return summary

    @staticmethod
    def _serialize_error(error: CrawlerError) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": error.error_class.value,
            "reason": error.reason,
        }
        if error.source:
            payload["source"] = error.source
        if error.provider:
            payload["provider"] = error.provider
        if error.status_code is not None:
            payload["status_code"] = error.status_code
        if error.attributes:
            payload["attributes"] = dict(error.attributes)
        return payload


__all__ = ["CrawlerWorker", "WorkerPublishResult"]
