"""Service helpers for composing crawler ingestion state."""

from __future__ import annotations

import base64
import hashlib
import re
import uuid
from datetime import timezone as datetime_timezone
from typing import Callable
from urllib.parse import urlsplit
from uuid import UUID, uuid4, uuid5

from django.utils import timezone
from rest_framework import status

from common.guardrails import FetcherLimits
from common.logging import get_logger
from crawler.contracts import normalize_source
from crawler.errors import ErrorClass
from crawler.fetcher import (
    FetchFailure,
    FetchRequest,
    FetchStatus,
    PolitenessContext,
)
from crawler.frontier import (
    CrawlSignals,
    FrontierAction,
    SourceDescriptor,
    decide_frontier_action,
)
from crawler.http_fetcher import HttpFetcher, HttpFetcherConfig
from documents.contract_utils import (
    normalize_media_type as normalize_document_media_type,
)
from documents.contracts import InlineBlob, NormalizedDocument
from documents.normalization import canonical_hash, normalize_url

from ai_core.contracts.crawler_runner import (
    CrawlerRunContext,
    CrawlerRunError,
    CrawlerStateBundle,
)
from ai_core.contracts.payloads import (
    FrontierData,
    GuardrailLimitsData,
    GuardrailPayload,
    GuardrailSignalsData,
)
from ai_core.infra.observability import emit_event
from ai_core.rag.guardrails import GuardrailLimits
from ai_core.telemetry.crawler import (
    build_fetch_payload,
    build_manual_fetch_payload,
    emit_fetch_started,
    record_fetch_attempt,
    summarize_fetch_attempt,
)

logger = get_logger(__name__)

FetcherFactory = Callable[[HttpFetcherConfig], HttpFetcher]


def build_crawler_state(
    context: CrawlerRunContext,
    *,
    fetcher_factory: FetcherFactory,
    lifecycle_store: object | None,
    object_store: object,
    guardrail_limits: GuardrailLimits | None = None,
) -> list[CrawlerStateBundle]:
    """Compose crawler graph state objects for each requested origin."""

    request_data = context.request
    workflow_id = context.workflow_id
    repository = context.repository
    meta = context.meta
    scope_meta = meta["scope_context"]
    # BREAKING CHANGE (Option A - Strict Separation):
    # Business IDs (case_id, workflow_id) now in business_context
    business_meta = meta.get("business_context", {})

    builds: list[CrawlerStateBundle] = []
    for origin in request_data.origins or []:
        provider = origin.provider or request_data.provider
        try:
            normalized_u = normalize_url(origin.url)
            source = normalize_source(provider, normalized_u or origin.url, None)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(str(exc)) from exc

        parsed = urlsplit(source.canonical_source)
        host = parsed.hostname or parsed.netloc
        if not host:
            raise ValueError("origin URL must include a valid host component")
        path_component = parsed.path or "/"

        descriptor = SourceDescriptor(
            host=host, path=path_component, provider=source.provider
        )
        frontier_data = FrontierData(
            host=descriptor.host,
            path=descriptor.path,
            provider=descriptor.provider,
            breadcrumbs=(),
            policy_events=(),
        )

        politeness = PolitenessContext(host=descriptor.host)
        fetch_request = FetchRequest(
            canonical_source=source.canonical_source, politeness=politeness
        )

        document_id = origin.document_id or request_data.document_id or uuid4().hex
        tags = tuple(_merge_origin_tags(request_data.tags, origin.tags))

        limit_bytes = _resolve_document_limit(origin, request_data, guardrail_limits)
        limits = GuardrailLimits(max_document_bytes=limit_bytes)

        snapshot_options = origin.snapshot
        if snapshot_options is None and (
            request_data.snapshot.enabled or request_data.snapshot.label
        ):
            snapshot_options = request_data.snapshot
        snapshot_requested = bool(snapshot_options and snapshot_options.enabled)
        snapshot_label = snapshot_options.label if snapshot_options else None

        dry_run = bool(
            origin.dry_run if origin.dry_run is not None else request_data.dry_run
        )
        review = origin.review or request_data.review or request_data.manual_review

        need_fetch = bool(origin.fetch or origin.content is None)
        body_bytes: bytes = b""
        effective_content_type = _normalize_media_type_value(
            origin.content_type or request_data.content_type
        )
        fetch_used = False
        http_status: int | None = None
        fetched_bytes: int | None = None
        fetch_elapsed: float | None = None
        fetch_retries: int | None = None
        fetch_retry_reason: str | None = None
        fetch_backoff_total_ms: float | None = None
        snapshot_path: str | None = None
        snapshot_sha256: str | None = None
        etag_value: str | None = None

        if need_fetch:
            decision = decide_frontier_action(descriptor, CrawlSignals())
            if decision.action is not FrontierAction.ENQUEUE:
                emit_event(
                    "crawler_robots_blocked",
                    {
                        "host": descriptor.host,
                        "reason": decision.reason,
                        "policy_events": list(decision.policy_events),
                    },
                )
                raise CrawlerRunError(
                    "Frontier denied the crawl due to robots or scheduling policies.",
                    code="crawler_robots_blocked",
                    status_code=status.HTTP_403_FORBIDDEN,
                    details={
                        "fetch_used": False,
                        "http_status": None,
                        "fetched_bytes": None,
                        "media_type_effective": None,
                        "fetch_elapsed": None,
                        "fetch_retries": None,
                        "fetch_retry_reason": None,
                        "fetch_backoff_total_ms": None,
                    },
                )

            emit_fetch_started(source.canonical_source, source.provider)
            fetch_limits = None
            if limits.max_document_bytes is not None:
                fetch_limits = FetcherLimits(max_bytes=limits.max_document_bytes)
            config = HttpFetcherConfig(limits=fetch_limits)
            fetcher = fetcher_factory(config)
            fetch_result = fetcher.fetch(fetch_request)
            record_fetch_attempt(fetch_result)

            if fetch_result.status is not FetchStatus.FETCHED:
                status_code, code = _map_fetch_error_response(fetch_result)
                emit_event(code, {"origin": source.canonical_source})
                failure_snapshot = summarize_fetch_attempt(
                    fetch_result,
                    media_type=_normalize_media_type_value(
                        fetch_result.metadata.content_type
                    ),
                )
                raise CrawlerRunError(
                    "Fetching the origin URL failed.",
                    code=code,
                    status_code=status_code,
                    details=failure_snapshot.as_details(),
                )

            fetch_used = True
            http_status = fetch_result.metadata.status_code
            effective_content_type = _normalize_media_type_value(
                fetch_result.metadata.content_type
            )
            failure = _map_failure(
                getattr(fetch_result.error, "error_class", None),
                getattr(fetch_result.error, "reason", None),
            )
            fetch_payload = build_fetch_payload(
                fetch_request,
                fetch_result,
                fetch_limits=fetch_limits,
                failure=failure,
                media_type=effective_content_type,
            )
            body_bytes = fetch_payload.body
            header_content_type = fetch_payload.headers.get("Content-Type")
            if header_content_type:
                effective_content_type = _normalize_media_type_value(
                    header_content_type
                )
            fetched_bytes = fetch_payload.downloaded_bytes
            fetch_elapsed = (
                fetch_payload.elapsed_ms / 1000 if fetch_payload.elapsed_ms else 0.0
            )
            fetch_retries = fetch_payload.retries
            fetch_retry_reason = fetch_payload.retry_reason
            fetch_backoff_total_ms = fetch_payload.backoff_total_ms
            etag_value = fetch_payload.headers.get("ETag")
            http_status = fetch_payload.status_code
        else:
            if origin.content is None:
                raise ValueError(
                    "Manual crawler runs require inline content. Provide content or enable remote fetching."
                )
            body_bytes = origin.content.encode("utf-8")
            manual_content_type = effective_content_type or "application/octet-stream"
            fetch_payload = build_manual_fetch_payload(
                fetch_request,
                body=body_bytes,
                media_type=manual_content_type,
            )
            fetched_bytes = fetch_payload.downloaded_bytes
            fetch_elapsed = fetch_payload.elapsed_ms / 1000
            fetch_retries = fetch_payload.retries
            fetch_retry_reason = fetch_payload.retry_reason
            fetch_backoff_total_ms = fetch_payload.backoff_total_ms
            http_status = fetch_payload.status_code
            etag_value = fetch_payload.headers.get("ETag")

            # Extract title from HTML if available and not already set
            if effective_content_type == "text/html" and body_bytes:
                try:
                    html_content = body_bytes.decode("utf-8", errors="ignore")
                    title_match = re.search(
                        r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL
                    )
                    if title_match:
                        extracted_title = title_match.group(1).strip()
                        if extracted_title:
                            # Only override if we don't have a better title
                            if not origin.title and not request_data.title:
                                origin.title = extracted_title
                except Exception:
                    pass

        if effective_content_type is None:
            effective_content_type = "application/octet-stream"

        guardrail_limits_data = GuardrailLimitsData(
            max_document_bytes=limits.max_document_bytes,
        )
        guardrail_signals = GuardrailSignalsData(
            tenant_id=str(scope_meta.get("tenant_id")),
            provider=source.provider,
            canonical_source=source.canonical_source,
            host=descriptor.host,
            document_bytes=len(body_bytes),
            mime_type=effective_content_type,
        )
        guardrail_payload = GuardrailPayload(
            decision="allow",
            reason="pending_evaluation",
            allowed=True,
            policy_events=(),
            limits=guardrail_limits_data,
            signals=guardrail_signals,
            attributes={},
        )

        fetched_at = timezone.now().astimezone(datetime_timezone.utc)
        document_uuid = _resolve_document_uuid(document_id)
        if document_uuid is None:
            document_uuid = uuid4()
        document_id = str(document_uuid)

        # collection_id should remain as string per ScopeContext contract
        collection_id_str = request_data.collection_id

        encoded_payload = base64.b64encode(body_bytes).decode("ascii")
        blob = InlineBlob(
            type="inline",
            media_type=effective_content_type,
            base64=encoded_payload,
            sha256=canonical_hash(body_bytes),
            size=len(body_bytes),
        )

        external_ref: dict[str, str] = {
            "provider": source.provider,
            "external_id": source.external_id,
        }
        original_requested_id = origin.document_id or request_data.document_id
        if original_requested_id:
            external_ref["crawler_document_id"] = str(original_requested_id)
        if etag_value:
            external_ref["etag"] = str(etag_value)

        normalized_document_input = NormalizedDocument(
            ref={
                "tenant_id": str(scope_meta.get("tenant_id")),
                "workflow_id": str(workflow_id),
                "document_id": document_uuid,
                "collection_id": collection_id_str,
            },
            meta={
                "tenant_id": str(scope_meta.get("tenant_id")),
                "workflow_id": str(workflow_id),
                "title": origin.title or request_data.title,
                "language": origin.language or request_data.language,
                "tags": list(tags),
                "origin_uri": source.canonical_source,
                "crawl_timestamp": fetched_at,
                "external_ref": external_ref,
            },
            blob=blob,
            checksum=blob.sha256,
            created_at=fetched_at,
            source="crawler",
        )

        if snapshot_requested and body_bytes:
            tenant_id = str(scope_meta.get("tenant_id"))
            # BREAKING CHANGE (Option A): case_id from business_context
            case_id = str(business_meta.get("case_id"))
            snapshot_path, snapshot_sha256 = _write_snapshot(
                object_store,
                tenant=tenant_id,
                case=case_id,
                payload=blob.decoded_payload(),
            )
        else:
            snapshot_path = None
            snapshot_sha256 = None

        state: dict[str, object] = {
            "tenant_id": scope_meta.get("tenant_id"),
            "case_id": business_meta.get(
                "case_id"
            ),  # BREAKING CHANGE (Option A): from business_context
            "workflow_id": workflow_id,
            "external_id": source.external_id,
            "origin_uri": source.canonical_source,
            "provider": source.provider,
            "frontier": frontier_data.model_dump(mode="json"),
            "fetch": fetch_payload.model_dump(mode="json"),
            "guardrails": guardrail_payload.model_dump(mode="json"),
            "document_id": document_id,
            "collection_id": request_data.collection_id,
            "normalized_document_input": normalized_document_input.model_dump(
                mode="json"
            ),
        }
        control: dict[str, object] = {
            "snapshot": snapshot_requested,
            "snapshot_label": snapshot_label,
            "fetch": fetch_used,
            "tags": list(tags),
            "shadow_mode": bool(request_data.shadow_mode or dry_run),
            "dry_run": dry_run,
            "mode": request_data.mode,
        }
        if review:
            control["review"] = review
            control["manual_review"] = review
        if request_data.force_retire:
            control["force_retire"] = True
        if request_data.recompute_delta:
            control["recompute_delta"] = True
        state["control"] = control

        baseline_data, previous_status = _load_baseline_context(
            scope_meta.get("tenant_id"),
            workflow_id,
            document_id,
            repository,
            lifecycle_store,
        )
        state["baseline"] = baseline_data
        if previous_status:
            state["previous_status"] = previous_status

        builds.append(
            CrawlerStateBundle(
                origin=source.canonical_source,
                provider=source.provider,
                document_id=document_id,
                state=state,
                fetch_used=fetch_used,
                http_status=http_status,
                fetched_bytes=fetched_bytes,
                media_type_effective=effective_content_type,
                fetch_elapsed=fetch_elapsed,
                fetch_retries=fetch_retries,
                fetch_retry_reason=fetch_retry_reason,
                fetch_backoff_total_ms=fetch_backoff_total_ms,
                snapshot_path=snapshot_path,
                snapshot_sha256=snapshot_sha256,
                tags=tags,
                collection_id=request_data.collection_id,
                snapshot_requested=snapshot_requested,
                snapshot_label=snapshot_label,
                review=review,
                dry_run=dry_run,
            )
        )

    return builds


def _resolve_document_limit(origin, request_data, defaults) -> int | None:
    if origin.limits and origin.limits.max_document_bytes is not None:
        return origin.limits.max_document_bytes
    if request_data.max_document_bytes is not None:
        return request_data.max_document_bytes
    if defaults and defaults.max_document_bytes is not None:
        return defaults.max_document_bytes
    return None


def _merge_origin_tags(global_tags, origin_tags) -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()
    for tag_list in (global_tags or []), (origin_tags or []):
        if not tag_list:
            continue
        for tag in tag_list:
            if not tag:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            combined.append(tag)
    return combined


def _normalize_media_type_value(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.split(";", 1)[0].strip()
    if not candidate:
        return None
    try:
        return normalize_document_media_type(candidate)
    except ValueError:
        return None


def _map_failure(error: ErrorClass | None, reason: str | None) -> FetchFailure | None:
    if error is None:
        return None
    if error is ErrorClass.TIMEOUT:
        return FetchFailure(reason=reason or "timeout", temporary=True)
    if error is ErrorClass.TRANSIENT_NETWORK:
        return FetchFailure(reason=reason or "network_error", temporary=True)
    if error is ErrorClass.RATE_LIMIT:
        return FetchFailure(reason=reason or "rate_limited", temporary=True)
    return FetchFailure(reason=reason or error.value, temporary=False)


def _map_fetch_error_response(result) -> tuple[int, str]:
    error = result.error
    if error is None:
        return status.HTTP_502_BAD_GATEWAY, "crawler_fetch_failed"

    error_class = getattr(error, "error_class", None)
    if error_class is ErrorClass.TIMEOUT:
        return status.HTTP_504_GATEWAY_TIMEOUT, "crawler_fetch_timeout"
    if error_class is ErrorClass.RATE_LIMIT:
        return status.HTTP_429_TOO_MANY_REQUESTS, "crawler_fetch_rate_limited"
    if error_class is ErrorClass.NOT_FOUND:
        return status.HTTP_404_NOT_FOUND, "crawler_fetch_not_found"
    if error_class is ErrorClass.GONE:
        return status.HTTP_410_GONE, "crawler_fetch_gone"
    if error_class is ErrorClass.POLICY_DENY:
        return status.HTTP_403_FORBIDDEN, "crawler_fetch_policy_denied"
    if error_class is ErrorClass.UPSTREAM_429:
        return status.HTTP_429_TOO_MANY_REQUESTS, "crawler_fetch_upstream_429"
    if error_class is ErrorClass.TRANSIENT_NETWORK:
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "crawler_fetch_transient_error",
        )
    return status.HTTP_502_BAD_GATEWAY, "crawler_fetch_failed"


def _write_snapshot(
    object_store, *, tenant: str, case: str, payload: bytes
) -> tuple[str, str]:
    sha256 = hashlib.sha256(payload).hexdigest()
    tenant_safe = object_store.sanitize_identifier(tenant)
    case_safe = object_store.sanitize_identifier(case)
    relative = "/".join((tenant_safe, case_safe, "crawler", f"{sha256}.html"))
    object_store.write_bytes(relative, payload)
    absolute = str(object_store.BASE_PATH / relative)
    return absolute, sha256


def _resolve_document_uuid(identifier: object) -> UUID | None:
    if isinstance(identifier, UUID):
        return identifier
    if identifier is None:
        return None
    try:
        candidate = str(identifier).strip()
    except Exception:  # pragma: no cover - defensive
        candidate = str(identifier)
    if not candidate:
        return None
    try:
        return UUID(candidate)
    except (TypeError, ValueError):
        return uuid5(uuid.NAMESPACE_URL, candidate)


def _load_baseline_context(
    tenant_id: object,
    workflow_id: object,
    document_identifier: object,
    repository: object | None,
    lifecycle_store: object | None,
) -> tuple[dict[str, object], str | None]:
    baseline: dict[str, object] = {}
    previous_status: str | None = None

    tenant: str | None = None
    if tenant_id is not None:
        tenant_candidate = str(tenant_id).strip()
        if tenant_candidate:
            tenant = tenant_candidate
    if not tenant:
        return baseline, previous_status

    document_uuid = _resolve_document_uuid(document_identifier)
    if document_uuid is None:
        return baseline, previous_status

    workflow: str | None = None
    if workflow_id is not None:
        workflow_candidate = str(workflow_id).strip()
        if workflow_candidate:
            workflow = workflow_candidate

    if repository is not None and hasattr(repository, "get"):
        try:
            existing = repository.get(  # type: ignore[attr-defined]
                tenant,
                document_uuid,
                prefer_latest=True,
                workflow_id=workflow,
            )
        except NotImplementedError:
            existing = None
        except Exception:  # pragma: no cover - best effort logging
            logger.debug(
                "crawler.baseline.repository_lookup_failed",
                extra={"tenant_id": tenant, "document_id": str(document_identifier)},
                exc_info=True,
            )
            existing = None

        if existing is not None:
            state = getattr(existing, "state", None)
            if state:
                baseline["normalized_document_state"] = state
            lifecycle_state = getattr(existing, "lifecycle_state", None)
            if lifecycle_state:
                lifecycle_text = str(lifecycle_state)
                baseline.setdefault("lifecycle_state", lifecycle_text)
                if previous_status is None:
                    previous_status = lifecycle_text

    if lifecycle_store is not None:
        getter = getattr(lifecycle_store, "get_document_state", None)
        if callable(getter):
            try:
                record = getter(  # type: ignore[misc]
                    tenant_id=tenant,
                    document_id=document_uuid,
                    workflow_id=workflow,
                )
            except Exception:  # pragma: no cover - best effort logging
                logger.debug(
                    "crawler.baseline.lifecycle_lookup_failed",
                    extra={
                        "tenant_id": tenant,
                        "document_id": str(document_identifier),
                    },
                    exc_info=True,
                )
                record = None

            if record is not None:
                state_value = getattr(record, "state", None)
                if state_value:
                    state_text = str(state_value)
                    baseline.setdefault("lifecycle_state", state_text)
                    previous_status = state_text
                reason_value = getattr(record, "reason", None)
                if reason_value:
                    baseline.setdefault("previous_reason", str(reason_value))
                events = getattr(record, "policy_events", None)
                if events:
                    baseline.setdefault("policy_events", tuple(events))

    return baseline, previous_status
