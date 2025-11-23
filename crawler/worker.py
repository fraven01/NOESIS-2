"""Crawler worker implementation forwarding fetch results to the AI Core."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple
from urllib.parse import urljoin, urlparse

from lxml import html

from ai_core.infra import object_store
from ai_core.infra.blob_writers import ObjectStoreBlobWriter
from ai_core.tasks import run_ingestion_graph
from common.assets import AssetIngestPayload, BlobWriter
from common.assets.hashing import perceptual_hash, sha256_bytes
from documents.contracts import DEFAULT_PROVIDER_BY_SOURCE

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
        ingestion_event_emitter: Optional[
            Callable[[str, Mapping[str, Any]], None]
        ] = None,
        blob_writer: Optional[BlobWriter] = None,
        blob_writer_factory: Optional[
            Callable[[str, Optional[str], Optional[str], Optional[str]], BlobWriter]
        ] = None,
    ) -> None:
        if ingestion_task is None:
            ingestion_task = run_ingestion_graph
        self._fetcher = fetcher
        self._ingestion_task = ingestion_task
        self._ingestion_event_emitter = ingestion_event_emitter
        if blob_writer is not None:
            self._blob_writer_factory = (
                blob_writer_factory
                or (lambda *_args, **_kwargs: blob_writer)
            )
        else:
            self._blob_writer_factory = (
                blob_writer_factory or self._default_blob_writer_factory
            )

    def process(
        self,
        request: FetchRequest,
        *,
        tenant_id: str,
        case_id: Optional[str] = None,
        crawl_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        trace_id: Optional[str] = None,
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

        propagated_trace_id = self._resolve_trace_id(trace_id, request)

        payload_state = self._compose_state(
            result,
            request,
            tenant_id=tenant_id,
            case_id=case_id,
            trace_id=propagated_trace_id,
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
            trace_id=propagated_trace_id,
            frontier_state=frontier_state,
            meta_overrides=meta_overrides,
        )
        if self._ingestion_event_emitter is not None:
            meta_payload.setdefault(
                "ingestion_event_emitter", self._ingestion_event_emitter
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
        trace_id: Optional[str],
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
        if trace_id is not None:
            state.setdefault("trace_id", trace_id)
        if crawl_id is not None:
            state.setdefault("crawl_id", crawl_id)
        state.setdefault("frontier", dict(frontier_state or {}))

        raw_meta: dict[str, Any] = dict(document_metadata or {})
        raw_meta.setdefault("origin_uri", request.canonical_source)

        resolved_source = self._resolve_source(raw_meta, ingestion_overrides)
        raw_meta["source"] = resolved_source

        provider_value = str(raw_meta.get("provider", "")).strip()
        if provider_value:
            raw_meta["provider"] = provider_value
        else:
            default_provider = DEFAULT_PROVIDER_BY_SOURCE.get(
                raw_meta["source"], raw_meta["source"]
            )
            if default_provider:
                raw_meta["provider"] = default_provider
        if result.metadata.content_type and "content_type" not in raw_meta:
            raw_meta["content_type"] = result.metadata.content_type
        if result.metadata.status_code is not None and "status_code" not in raw_meta:
            raw_meta["status_code"] = result.metadata.status_code
        if result.metadata.etag and "etag" not in raw_meta:
            raw_meta["etag"] = result.metadata.etag
        if result.metadata.last_modified and "last_modified" not in raw_meta:
            raw_meta["last_modified"] = result.metadata.last_modified
        if (
            result.metadata.content_length is not None
            and "content_length" not in raw_meta
        ):
            raw_meta["content_length"] = result.metadata.content_length

        payload_bytes = bytes(result.payload or b"")
        payload_path, payload_checksum, payload_size = self._persist_payload(
            payload_bytes,
            tenant_id=tenant_id,
            case_id=case_id,
            crawl_id=crawl_id,
            document_id=document_id or raw_meta.get("document_id"),
        )

        raw_meta.setdefault("content_hash", payload_checksum)
        raw_meta.setdefault("content_length", payload_size)

        raw_document: dict[str, Any] = {
            "metadata": raw_meta,
            "payload_path": payload_path,
        }
        if document_id is not None:
            raw_document["document_id"] = document_id
        elif "document_id" in raw_meta:
            raw_document["document_id"] = raw_meta["document_id"]

        state["raw_document"] = raw_document
        state.setdefault("raw_payload_path", payload_path)
        state.setdefault("fetch", self._summarize_fetch(result))
        assets = self._extract_assets(
            result,
            payload_bytes,
            tenant_id=tenant_id,
            case_id=case_id,
            crawl_id=crawl_id,
            document_id=document_id or raw_meta.get("document_id"),
        )
        if assets:
            state["assets"] = [self._serialize_asset(asset) for asset in assets]
        return state

    def _resolve_source(
        self,
        metadata: Mapping[str, Any],
        ingestion_overrides: Optional[Mapping[str, Any]],
    ) -> str:
        candidate = self._normalize_source_value(metadata.get("source"))
        if candidate:
            return candidate

        if ingestion_overrides:
            override_candidate = self._normalize_source_value(
                ingestion_overrides.get("source")
            )
            if override_candidate:
                return override_candidate
            raw_document = ingestion_overrides.get("raw_document")
            if isinstance(raw_document, Mapping):
                metadata_override = raw_document.get("metadata")
                if isinstance(metadata_override, Mapping):
                    nested_candidate = self._normalize_source_value(
                        metadata_override.get("source")
                    )
                    if nested_candidate:
                        return nested_candidate

        raise ValueError("document_metadata.source_required")

    @staticmethod
    def _normalize_source_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _persist_payload(
        self,
        payload: bytes,
        *,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        document_id: Optional[str],
    ) -> Tuple[str, str, int]:
        writer = self._blob_writer_for(
            tenant_id=tenant_id,
            case_id=case_id,
            crawl_id=crawl_id,
            document_id=document_id,
            scope="raw",
        )
        uri, checksum, size = writer.put(payload)
        return uri, checksum, size

    def _blob_writer_for(
        self,
        *,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        document_id: Optional[str],
        scope: str,
    ) -> BlobWriter:
        try:
            return self._blob_writer_factory(
                tenant_id, case_id, crawl_id, document_id, scope=scope
            )
        except TypeError:
            if scope != "raw":
                return self._default_blob_writer_factory(
                    tenant_id, case_id, crawl_id, document_id, scope=scope
                )
            return self._blob_writer_factory(
                tenant_id, case_id, crawl_id, document_id
            )

    def _default_blob_writer_factory(
        self,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        document_id: Optional[str],
        *,
        scope: str = "raw",
    ) -> BlobWriter:
        safe_tenant = self._safe_identifier(tenant_id, "tenant")
        safe_case = self._safe_identifier(case_id, "case")
        safe_crawl = self._safe_identifier(crawl_id, "crawl") if crawl_id else None
        safe_document = (
            self._safe_identifier(document_id, "document") if document_id else None
        )

        path_parts = [safe_tenant, safe_case, "crawler", scope]
        if safe_crawl:
            path_parts.append(safe_crawl)
        if safe_document:
            path_parts.append(safe_document)
        return ObjectStoreBlobWriter(prefix=path_parts)

    @staticmethod
    def _safe_identifier(value: Optional[str], prefix: str) -> str:
        if value:
            candidate = str(value).strip()
            if candidate:
                try:
                    return object_store.sanitize_identifier(candidate)
                except ValueError:
                    pass
        fallback = f"{prefix}-{hashlib.sha256(str(value or '').encode('utf-8')).hexdigest()[:12]}"
        return object_store.sanitize_identifier(fallback)

    def _compose_meta(
        self,
        *,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        idempotency_key: Optional[str],
        trace_id: Optional[str],
        frontier_state: Optional[Mapping[str, Any]],
        meta_overrides: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tenant_id": tenant_id,
            "case_id": case_id,
            "crawl_id": crawl_id,
            "idempotency_key": idempotency_key,
            "trace_id": trace_id,
        }
        if frontier_state:
            payload.setdefault("frontier", dict(frontier_state))
        if meta_overrides:
            payload.update(dict(meta_overrides))
        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _resolve_trace_id(
        provided_trace_id: Optional[str], request: FetchRequest
    ) -> Optional[str]:
        if provided_trace_id:
            candidate = str(provided_trace_id).strip()
            if candidate:
                return candidate
        metadata_trace = request.metadata.get("trace_id")
        if metadata_trace is None:
            return None
        candidate = str(metadata_trace).strip()
        return candidate or None

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

    @staticmethod
    def _serialize_asset(asset: AssetIngestPayload) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "media_type": asset.media_type,
            "metadata": dict(asset.metadata),
        }
        if asset.content is not None:
            payload["content"] = base64.b64encode(asset.content).decode("ascii")
        if asset.file_uri is not None:
            payload["file_uri"] = asset.file_uri
        if asset.page_index is not None:
            payload["page_index"] = asset.page_index
        if asset.bbox is not None:
            payload["bbox"] = list(asset.bbox)
        if asset.context_before is not None:
            payload["context_before"] = asset.context_before
        if asset.context_after is not None:
            payload["context_after"] = asset.context_after
        return payload

    def _extract_assets(
        self,
        result: FetchResult,
        payload: bytes,
        *,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        document_id: Optional[str],
    ) -> list[AssetIngestPayload]:
        content_type = (result.metadata.content_type or "").lower()
        if "html" not in content_type:
            return []

        try:
            tree = html.fromstring(payload)
        except Exception:
            return []

        base_url = result.request.canonical_source
        assets: list[AssetIngestPayload] = []
        seen_urls: set[str] = set()
        for element in self._iter_image_elements(tree):
            raw_src = element.get("src") or element.get("data-src")
            normalized = self._normalize_asset_url(raw_src, base_url)
            if normalized is None:
                continue

            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            asset_bytes, media_type = self._download_asset(normalized, result.request)
            if asset_bytes is None or media_type is None:
                continue

            sha256 = sha256_bytes(asset_bytes)
            perceptual = perceptual_hash(asset_bytes)
            locator = self._asset_locator(normalized)
            file_uri = self._persist_asset_blob(
                asset_bytes,
                locator,
                sha256,
                tenant_id=tenant_id,
                case_id=case_id,
                crawl_id=crawl_id,
                document_id=document_id,
            )
            caption_candidates = []
            alt_text = (element.get("alt") or "").strip()
            if alt_text:
                caption_candidates.append(("alt_text", alt_text))

            metadata: dict[str, Any] = {
                "origin_uri": normalized,
                "locator": locator,
                "caption_candidates": caption_candidates,
                "sha256": sha256,
            }
            if perceptual:
                metadata["perceptual_hash"] = perceptual

            assets.append(
                AssetIngestPayload(
                    media_type=media_type,
                    metadata=metadata,
                    file_uri=file_uri,
                    content=None,
                )
            )

        return assets

    @staticmethod
    def _iter_image_elements(tree: html.HtmlElement) -> Iterable[html.HtmlElement]:
        return tree.iterfind(".//img")

    def _normalize_asset_url(self, raw: Optional[str], base_url: str) -> Optional[str]:
        if raw is None:
            return None
        candidate = raw.strip()
        if not candidate:
            return None
        joined = urljoin(base_url, candidate)
        parsed = urlparse(joined)
        if parsed.scheme.lower() not in {"http", "https"}:
            return None
        return joined

    def _download_asset(
        self, url: str, request: FetchRequest
    ) -> tuple[Optional[bytes], Optional[str]]:
        try:
            fetch_result = self._fetcher.fetch(
                FetchRequest(
                    canonical_source=url,
                    politeness=request.politeness,
                    metadata=request.metadata,
                )
            )
        except Exception:
            return None, None

        if fetch_result.status is not FetchStatus.FETCHED or not fetch_result.payload:
            return None, None

        media_type = fetch_result.metadata.content_type
        if not media_type:
            guessed, _ = mimetypes.guess_type(url)
            media_type = guessed or "application/octet-stream"
        return bytes(fetch_result.payload), media_type

    def _asset_locator(self, url: str) -> str:
        parsed = urlparse(url)
        filename = parsed.path.rsplit("/", 1)[-1]
        if filename:
            try:
                return object_store.safe_filename(filename)
            except Exception:
                pass
        return parsed.hostname or "asset"

    def _persist_asset_blob(
        self,
        payload: bytes,
        locator: str,
        sha256: str,
        *,
        tenant_id: str,
        case_id: Optional[str],
        crawl_id: Optional[str],
        document_id: Optional[str],
        ) -> str:
        writer = self._blob_writer_for(
            tenant_id=tenant_id,
            case_id=case_id,
            crawl_id=crawl_id,
            document_id=document_id,
            scope="assets",
        )
        uri, *_rest = writer.put(payload)
        return uri


__all__ = ["CrawlerWorker", "WorkerPublishResult"]
