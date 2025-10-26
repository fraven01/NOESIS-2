from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from crawler.contracts import NormalizedSource, normalize_source
from crawler.delta import DeltaDecision, DeltaStatus, evaluate_delta
from crawler.errors import ErrorClass
from crawler.fetcher import (
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetcherLimits,
    PolitenessContext,
    evaluate_fetch_response,
)
from crawler.frontier import (
    CrawlSignals,
    FrontierDecision,
    RobotsPolicy,
    SourceDescriptor,
    decide_frontier_action,
)
from crawler.ingestion import (
    IngestionDecision,
    IngestionStatus,
    build_ingestion_decision,
)
from crawler.normalizer import NormalizedDocument, build_normalized_document
from crawler.parser import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    StructuralElement,
    build_parse_result,
    compute_parser_stats,
)
from crawler.retire import evaluate_lifecycle

TENANT_ID = "tenant-main"
WORKFLOW_ID = "wf-e2e"
CASE_ID = "case-007"


@dataclass(frozen=True)
class SpanEvent:
    name: str
    attributes: Mapping[str, object]


@dataclass(frozen=True)
class SpanSnapshot:
    name: str
    attributes: Mapping[str, object]
    events: Tuple[SpanEvent, ...] = ()


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    external_id: str
    provider: str
    tenant_id: str
    workflow_id: str
    document_id: Optional[str]
    source_kind: str = "http"

    def annotate(self, attributes: Mapping[str, object]) -> Dict[str, object]:
        base: Dict[str, object] = {
            "crawler.trace_id": self.trace_id,
            "noesis.external_id": self.external_id,
            "crawler.provider": self.provider,
            "crawler.source_kind": self.source_kind,
        }
        if self.tenant_id:
            base["noesis.tenant_id"] = self.tenant_id
        if self.workflow_id:
            base["noesis.workflow_id"] = self.workflow_id
        if self.document_id:
            base["noesis.document_id"] = self.document_id
        merged = dict(base)
        merged.update(attributes)
        return merged


def test_tracking_parameter_explosion_produces_stable_external_id() -> None:
    noisy_url = "https://Example.com/articles?id=42&utm_source=newsletter&A=1"
    tracker_heavy = (
        "https://example.com/articles?id=42&A=1&ICID=mid-banner&utm_medium=email#frag"
    )

    first_source = normalize_source("web", noisy_url, None)
    second_source = normalize_source("web", tracker_heavy, None)

    assert first_source.external_id == second_source.external_id
    assert first_source.canonical_source == second_source.canonical_source

    trace = TraceContext(
        trace_id="trace-tracking",
        external_id=first_source.external_id,
        provider=first_source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-tracking",
    )

    descriptor = _descriptor_from_source(first_source)
    frontier = decide_frontier_action(descriptor)
    fetch = _make_fetch_result(
        first_source.canonical_source, body=b"<html>Hello</html>", etag="v1"
    )
    parse = _make_parse_result(
        fetch, primary_text="Hello world", title="Hello", language="en"
    )
    document = _make_document(parse, first_source, document_id="doc-tracking")
    delta = evaluate_delta(document)
    ingestion = build_ingestion_decision(document, delta, case_id=CASE_ID)

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
        "fetch.http": _build_fetch_span(trace, fetch),
        "parse": _build_parse_span(trace, parse),
        "normalize": _build_normalize_span(trace, document),
        "delta": _build_delta_span(trace, delta, None, ingestion.status, CASE_ID),
        "ingest": _build_ingest_span(trace, ingestion),
    }

    assert delta.status is DeltaStatus.NEW
    assert ingestion.status is IngestionStatus.UPSERT
    assert {
        "frontier.queue",
        "fetch.http",
        "parse",
        "normalize",
        "delta",
        "ingest",
    } <= set(spans)
    assert spans["fetch.http"].attributes["http.status_code"] == 200
    assert spans["delta"].events and spans["delta"].events[0].name == "changed"


def test_not_modified_chain_results_in_unchanged_delta() -> None:
    url = "https://example.com/docs/reference"
    baseline_source, _, _, document, baseline_delta = _build_document_state(
        url, text="Original reference", document_id="doc-ref"
    )

    previous_hash = baseline_delta.signatures.content_hash
    previous_version = baseline_delta.version

    trace = TraceContext(
        trace_id="trace-304",
        external_id=baseline_source.external_id,
        provider=baseline_source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-ref",
    )

    descriptor = _descriptor_from_source(baseline_source)
    frontier = decide_frontier_action(
        descriptor,
        signals=CrawlSignals(last_crawled_at=datetime.now(timezone.utc)),
    )

    fetch = _make_fetch_result(
        baseline_source.canonical_source,
        status_code=304,
        body=None,
        etag="v1",
        downloaded_bytes=0,
    )
    parse = _make_parse_result(
        fetch,
        primary_text="Original reference",
        title="Reference",
        language="en",
    )
    document_repeat = _make_document(parse, baseline_source, document_id="doc-ref")
    delta = evaluate_delta(
        document_repeat,
        previous_content_hash=previous_hash,
        previous_version=previous_version,
    )
    ingestion = build_ingestion_decision(document_repeat, delta, case_id=CASE_ID)

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
        "fetch.http": _build_fetch_span(trace, fetch),
        "parse": _build_parse_span(trace, parse),
        "normalize": _build_normalize_span(trace, document_repeat),
        "delta": _build_delta_span(
            trace, delta, previous_hash, ingestion.status, CASE_ID
        ),
        "ingest": _build_ingest_span(trace, ingestion),
    }

    assert fetch.status is FetchStatus.NOT_MODIFIED
    assert any(event.name == "not_modified" for event in spans["fetch.http"].events)
    assert delta.status is DeltaStatus.UNCHANGED
    assert ingestion.status is IngestionStatus.SKIP
    assert spans["delta"].events[0].name == "unchanged"


def test_content_change_advances_version_and_upserts() -> None:
    url = "https://example.com/news/story"
    source, _, _, previous_document, previous_delta = _build_document_state(
        url, text="Breaking news", document_id="doc-story"
    )

    previous_hash = previous_delta.signatures.content_hash
    previous_version = previous_delta.version

    trace = TraceContext(
        trace_id="trace-change",
        external_id=source.external_id,
        provider=source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-story",
    )

    descriptor = _descriptor_from_source(source)
    frontier = decide_frontier_action(descriptor)

    updated_body = b"<html><body>Breaking news update</body></html>"
    fetch = _make_fetch_result(source.canonical_source, body=updated_body, etag="v2")
    parse = _make_parse_result(
        fetch,
        primary_text="Breaking news update",
        title="Breaking",
        language="en",
    )
    document_updated = _make_document(parse, source, document_id="doc-story")
    delta = evaluate_delta(
        document_updated,
        previous_content_hash=previous_hash,
        previous_version=previous_version,
    )
    ingestion = build_ingestion_decision(document_updated, delta, case_id=CASE_ID)

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
        "fetch.http": _build_fetch_span(trace, fetch),
        "parse": _build_parse_span(trace, parse),
        "normalize": _build_normalize_span(trace, document_updated),
        "delta": _build_delta_span(
            trace, delta, previous_hash, ingestion.status, CASE_ID
        ),
        "ingest": _build_ingest_span(trace, ingestion),
    }

    assert delta.status is DeltaStatus.CHANGED
    assert delta.version == (previous_version or 0) + 1
    assert ingestion.status is IngestionStatus.UPSERT
    assert spans["delta"].events[0].name == "changed"


def test_robots_deny_shortcircuits_fetch() -> None:
    source = normalize_source("web", "https://example.com/restricted/area", None)
    trace = TraceContext(
        trace_id="trace-robots",
        external_id=source.external_id,
        provider=source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-robots",
    )

    descriptor = _descriptor_from_source(source)
    robots = RobotsPolicy(disallow=("/restricted",))
    frontier = decide_frontier_action(descriptor, robots=robots)

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
    }

    assert frontier.reason == "robots_disallow"
    assert frontier.action.value == "skip"
    assert not frontier.earliest_visit_at
    assert any(event.name == "policy_deny" for event in spans["frontier.queue"].events)
    assert "fetch.http" not in spans


def test_unsupported_media_emits_parser_error() -> None:
    source = normalize_source("web", "https://example.com/archive/data.bin", None)
    trace = TraceContext(
        trace_id="trace-unsupported",
        external_id=source.external_id,
        provider=source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-unsupported",
    )

    descriptor = _descriptor_from_source(source)
    frontier = decide_frontier_action(descriptor)
    fetch = _make_fetch_result(
        source.canonical_source,
        body=b"%PDF-1.7",
        etag="pdf",
        content_type="application/pdf",
    )
    stats = compute_parser_stats(primary_text=None, extraction_path="pdf.binary")
    parse = build_parse_result(
        fetch,
        status=ParseStatus.UNSUPPORTED_MEDIA,
        stats=stats,
        diagnostics=["unsupported pdf"],
    )

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
        "fetch.http": _build_fetch_span(trace, fetch),
        "parse": _build_parse_span(trace, parse),
    }

    assert parse.error is not None
    assert parse.error.error_class is ErrorClass.UNSUPPORTED_MEDIA
    assert (
        spans["parse"].attributes["crawler.error_class"]
        == ErrorClass.UNSUPPORTED_MEDIA.value
    )
    assert (
        spans["parse"].events and spans["parse"].events[0].name == "unsupported_media"
    )
    assert "normalize" not in spans
    assert "delta" not in spans


def test_near_duplicate_detected_without_upsert() -> None:
    url = "https://example.com/blog/post"
    base_source, _, _, base_document, base_delta = _build_document_state(
        url, text="Insightful analysis", document_id="doc-original"
    )

    trace = TraceContext(
        trace_id="trace-duplicate",
        external_id=base_source.external_id,
        provider=base_source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-duplicate",
    )

    descriptor = _descriptor_from_source(base_source)
    frontier = decide_frontier_action(descriptor)

    duplicate_fetch = _make_fetch_result(
        base_source.canonical_source,
        body=b"<html>Insightful analysis!</html>",
        etag="dup",
    )
    duplicate_parse = _make_parse_result(
        duplicate_fetch,
        primary_text="Insightful analysis!",
        title="Insight",
        language="en",
    )
    duplicate_source = NormalizedSource(
        provider=base_source.provider,
        canonical_source=base_source.canonical_source,
        external_id=base_source.external_id,
        provider_tags=base_source.provider_tags,
    )
    duplicate_document = _make_document(
        duplicate_parse,
        duplicate_source,
        document_id="doc-duplicate",
    )
    assert base_delta.signatures.near_duplicate is not None
    known_signatures = {
        "doc-original": base_delta.signatures.near_duplicate,
    }
    delta = evaluate_delta(
        duplicate_document,
        previous_content_hash=None,
        known_near_duplicates=known_signatures,
    )
    ingestion = build_ingestion_decision(duplicate_document, delta, case_id=CASE_ID)

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
        "fetch.http": _build_fetch_span(trace, duplicate_fetch),
        "parse": _build_parse_span(trace, duplicate_parse),
        "normalize": _build_normalize_span(trace, duplicate_document),
        "delta": _build_delta_span(trace, delta, None, ingestion.status, CASE_ID),
        "ingest": _build_ingest_span(trace, ingestion),
    }

    assert delta.status is DeltaStatus.NEAR_DUPLICATE
    assert ingestion.status is IngestionStatus.SKIP
    assert spans["delta"].events[0].name == "near_duplicate"
    assert spans["ingest"].attributes["ingest.decision"] == "skip"


def test_gone_fetch_triggers_retire_lifecycle() -> None:
    url = "https://example.com/catalog/item"
    source, _, _, document, base_delta = _build_document_state(
        url, text="Catalog entry", document_id="doc-item"
    )

    trace = TraceContext(
        trace_id="trace-retire",
        external_id=source.external_id,
        provider=source.provider,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id="doc-item",
    )

    descriptor = _descriptor_from_source(source)
    frontier = decide_frontier_action(descriptor)
    gone_fetch = _make_fetch_result(
        source.canonical_source,
        status_code=410,
        body=None,
        etag="gone",
        downloaded_bytes=0,
    )
    lifecycle = evaluate_lifecycle(fetch=gone_fetch)
    delta = DeltaDecision(
        status=DeltaStatus.UNCHANGED,
        signatures=base_delta.signatures,
        version=base_delta.version,
        reason="hash_match",
    )
    ingestion = build_ingestion_decision(
        document,
        delta,
        case_id=CASE_ID,
        lifecycle=lifecycle,
    )

    spans = {
        "frontier.queue": _build_frontier_span(
            trace, descriptor, frontier, CrawlSignals()
        ),
        "fetch.http": _build_fetch_span(trace, gone_fetch),
        "delta": _build_delta_span(
            trace, delta, delta.signatures.content_hash, ingestion.status, CASE_ID
        ),
        "ingest": _build_ingest_span(trace, ingestion),
    }

    assert gone_fetch.status is FetchStatus.GONE
    assert gone_fetch.error is not None
    assert gone_fetch.error.error_class is ErrorClass.GONE
    assert ingestion.status is IngestionStatus.RETIRE
    assert lifecycle.policy_events == ("gone_410",)
    assert spans["fetch.http"].events[0].name == "gone"
    assert spans["ingest"].attributes["ingest.decision"] == "retire"


def _build_document_state(
    url: str,
    *,
    text: str,
    document_id: str,
    etag: str = "v1",
    last_modified: Optional[str] = None,
) -> Tuple[
    NormalizedSource, FetchResult, ParseResult, NormalizedDocument, DeltaDecision
]:
    source = normalize_source("web", url, None)
    fetch = _make_fetch_result(
        source.canonical_source,
        body=text.encode("utf-8"),
        etag=etag,
        last_modified=last_modified,
    )
    parse = _make_parse_result(
        fetch,
        primary_text=text,
        title="Title",
        language="en",
    )
    document = _make_document(parse, source, document_id=document_id)
    delta = evaluate_delta(document)
    return source, fetch, parse, document, delta


def _descriptor_from_source(source: NormalizedSource) -> SourceDescriptor:
    parts = urlsplit(source.canonical_source)
    host = parts.hostname or ""
    path = parts.path or "/"
    return SourceDescriptor(host=host, path=path, provider=source.provider)


def _make_fetch_result(
    url: str,
    *,
    status_code: int = 200,
    body: Optional[bytes],
    etag: Optional[str],
    last_modified: Optional[str] = None,
    content_type: str = "text/html; charset=utf-8",
    retries: int = 0,
    downloaded_bytes: Optional[int] = None,
) -> FetchResult:
    request = FetchRequest(
        canonical_source=url,
        politeness=PolitenessContext(
            host=(urlsplit(url).hostname or "example.com"), user_agent="CrawlerTest/1.0"
        ),
        metadata={"provider": "web"},
    )
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    if etag:
        headers["ETag"] = etag
    if last_modified:
        headers["Last-Modified"] = last_modified
    elapsed = 0.2
    if downloaded_bytes is None and body is not None:
        downloaded_bytes = len(body)
    limits = FetcherLimits()
    return evaluate_fetch_response(
        request,
        status_code=status_code,
        body=body,
        headers=headers,
        elapsed=elapsed,
        retries=retries,
        limits=limits,
        downloaded_bytes=downloaded_bytes,
    )


def _make_parse_result(
    fetch: FetchResult,
    *,
    primary_text: str,
    title: Optional[str],
    language: Optional[str],
) -> ParseResult:
    content = ParserContent(
        media_type=(fetch.metadata.content_type or "text/html").split(";")[0],
        primary_text=primary_text,
        title=title,
        content_language=language,
        structural_elements=(
            StructuralElement(kind="heading", text=title or primary_text[:30]),
            StructuralElement(kind="paragraph", text=primary_text),
        ),
    )
    stats = ParserStats(
        token_count=len(primary_text.split()),
        character_count=len(primary_text),
        extraction_path="html.body",
        warnings=("ok",) if primary_text else (),
    )
    return build_parse_result(
        fetch,
        status=ParseStatus.PARSED,
        content=content,
        stats=stats,
        diagnostics=["parsed"],
    )


def _make_document(
    parse: ParseResult,
    source: NormalizedSource,
    *,
    document_id: str,
) -> NormalizedDocument:
    return build_normalized_document(
        parse_result=parse,
        source=source,
        tenant_id=TENANT_ID,
        workflow_id=WORKFLOW_ID,
        document_id=document_id,
        tags=("news",),
    )


def _build_frontier_span(
    trace: TraceContext,
    descriptor: SourceDescriptor,
    decision: FrontierDecision,
    signals: CrawlSignals,
) -> SpanSnapshot:
    earliest = (
        decision.earliest_visit_at.isoformat() if decision.earliest_visit_at else None
    )
    attributes = trace.annotate(
        {
            "crawler.host": descriptor.host,
            "crawler.path": descriptor.path,
            "crawler.policy_decision": decision.action.value,
            "crawler.next_visit_at": earliest,
            "crawler.retry_count": signals.consecutive_failures,
            "crawler.crawl_delay_ms": (
                descriptor.metadata.get("crawl_delay") if descriptor.metadata else None
            ),
            "crawler.manual_override": signals.override_recrawl_frequency is not None,
            "noesis.reason": decision.reason,
        }
    )
    events = []
    if "robots_disallow" in decision.policy_events:
        events.append(SpanEvent("policy_deny", {"reason": "robots_disallow"}))
    if "robots_allow" in decision.policy_events:
        events.append(SpanEvent("policy_allow", {}))
    return SpanSnapshot(
        name="frontier.queue", attributes=attributes, events=tuple(events)
    )


def _build_fetch_span(trace: TraceContext, fetch: FetchResult) -> SpanSnapshot:
    parts = urlsplit(fetch.request.canonical_source)
    latency = (
        fetch.telemetry.latency * 1000 if fetch.telemetry.latency is not None else None
    )
    policy_state = "ok"
    if fetch.status is FetchStatus.POLICY_DENIED:
        policy_state = "denied"
    elif fetch.status is FetchStatus.TEMPORARY_ERROR and fetch.detail:
        policy_state = fetch.detail
    attributes = trace.annotate(
        {
            "http.status_code": fetch.metadata.status_code,
            "http.host": parts.hostname or "",
            "http.method": "GET",
            "http.user_agent": fetch.request.politeness.user_agent,
            "http.retry": fetch.telemetry.retries,
            "http.latency_ms": latency,
            "http.response_size": fetch.telemetry.bytes_downloaded,
            "http.content_type": fetch.metadata.content_type,
            "http.etag": fetch.metadata.etag,
            "http.last_modified": fetch.metadata.last_modified,
            "crawler.policy_enforced": policy_state,
        }
    )
    if fetch.error is not None:
        attributes["crawler.error_class"] = fetch.error.error_class.value
    events: list[SpanEvent] = []
    if fetch.status is FetchStatus.NOT_MODIFIED:
        events.append(
            SpanEvent(
                "not_modified",
                {
                    "etag": fetch.metadata.etag,
                    "last_modified": fetch.metadata.last_modified,
                },
            )
        )
    if fetch.status is FetchStatus.GONE:
        events.append(
            SpanEvent(
                "gone",
                {
                    "status_code": fetch.metadata.status_code,
                    "reason": fetch.detail,
                },
            )
        )
    if fetch.status is FetchStatus.TEMPORARY_ERROR:
        events.append(
            SpanEvent(
                "temporary_error",
                {
                    "status_code": fetch.metadata.status_code,
                    "reason": fetch.detail,
                },
            )
        )
    return SpanSnapshot(name="fetch.http", attributes=attributes, events=tuple(events))


def _build_parse_span(trace: TraceContext, parse: ParseResult) -> SpanSnapshot:
    fetch = parse.fetch
    bytes_in = fetch.telemetry.bytes_downloaded
    primary_length = (
        len(parse.content.primary_text) if parse.status is ParseStatus.PARSED else 0
    )
    token_count = parse.stats.token_count if parse.status is ParseStatus.PARSED else 0
    language = (
        parse.content.content_language if parse.status is ParseStatus.PARSED else None
    )
    warning_count = (
        len(parse.stats.warnings)
        if parse.status is ParseStatus.PARSED and parse.stats
        else 0
    )
    attributes = trace.annotate(
        {
            "parser.media_type": (
                parse.content.media_type
                if parse.content
                else fetch.metadata.content_type
            ),
            "parser.bytes_in": bytes_in,
            "parser.primary_text_length": primary_length,
            "parser.token_count": token_count,
            "parser.language": language,
            "parser.warning_count": warning_count,
            "parser.strategy": parse.stats.extraction_path if parse.stats else None,
        }
    )
    events: list[SpanEvent] = []
    if parse.error is not None:
        attributes["crawler.error_class"] = parse.error.error_class.value
        events.append(
            SpanEvent(
                parse.status.value,
                {
                    "reason": parse.error.reason,
                    "error_class": parse.error.error_class.value,
                },
            )
        )
    return SpanSnapshot(name="parse", attributes=attributes, events=tuple(events))


def _build_normalize_span(
    trace: TraceContext, document: NormalizedDocument
) -> SpanSnapshot:
    parser_stats = document.meta.parser_stats
    parser_warnings = parser_stats.get("parser.warnings", [])
    warning_count = len(parser_warnings) if isinstance(parser_warnings, Sequence) else 0
    attributes = trace.annotate(
        {
            "normalizer.origin_uri": document.meta.origin_uri,
            "normalizer.title": document.meta.title,
            "normalizer.language": document.meta.language,
            "normalizer.bytes_in": parser_stats.get("normalizer.bytes_in"),
            "normalizer.tag_count": len(document.meta.tags),
            "normalizer.parser_token_count": document.stats.token_count,
            "normalizer.validation_status": "ok",
            "normalizer.warning_count": len(document.diagnostics) + warning_count,
        }
    )
    return SpanSnapshot(name="normalize", attributes=attributes, events=())


def _build_delta_span(
    trace: TraceContext,
    delta: DeltaDecision,
    previous_hash: Optional[str],
    ingestion_status: IngestionStatus,
    case_id: str,
) -> SpanSnapshot:
    score = None
    if delta.reason.startswith("near_duplicate:"):
        try:
            score = float(delta.reason.split(":", 1)[1])
        except ValueError:
            score = None
    attributes = trace.annotate(
        {
            "delta.status": delta.status.value,
            "delta.content_hash": delta.signatures.content_hash[:8],
            "delta.previous_hash": previous_hash[:8] if previous_hash else None,
            "delta.version": delta.version,
            "delta.near_duplicate_score": score,
            "delta.reference_document_id": delta.parent_document_id,
            "delta.policy_action": _policy_action_from_ingestion(ingestion_status),
            "noesis.case_id": case_id,
        }
    )
    events: list[SpanEvent] = []
    if delta.status in {DeltaStatus.NEW, DeltaStatus.CHANGED}:
        events.append(
            SpanEvent(
                "changed",
                {
                    "content_hash": delta.signatures.content_hash,
                    "version": delta.version,
                },
            )
        )
    elif delta.status is DeltaStatus.UNCHANGED:
        events.append(
            SpanEvent(
                "unchanged",
                {
                    "content_hash": delta.signatures.content_hash,
                },
            )
        )
    elif delta.status is DeltaStatus.NEAR_DUPLICATE:
        events.append(
            SpanEvent(
                "near_duplicate",
                {
                    "score": score,
                    "reference_document_id": delta.parent_document_id,
                },
            )
        )
    return SpanSnapshot(name="delta", attributes=attributes, events=tuple(events))


def _build_ingest_span(
    trace: TraceContext, decision: IngestionDecision
) -> SpanSnapshot:
    payload_size = 0
    chunk_count = 0
    if decision.payload is not None:
        payload_size = _estimate_payload_size(decision.payload)
        chunk_count = 1 if decision.payload.media_type else 0
    attributes = trace.annotate(
        {
            "ingest.decision": decision.status.value,
            "ingest.chunk_count": chunk_count,
            "ingest.payload_size": payload_size,
            "ingest.embedding_profile": None,
            "ingest.queue": "ingestion",
            "ingest.retries": 0,
            "ingest.error_code": None,
            "noesis.case_id": decision.payload.case_id if decision.payload else CASE_ID,
        }
    )
    if decision.policy_events:
        attributes["ingest.policy_events"] = decision.policy_events
    return SpanSnapshot(name="ingest", attributes=attributes, events=())


def _estimate_payload_size(payload) -> int:
    payload_dict = {
        "tenant_id": payload.tenant_id,
        "case_id": payload.case_id,
        "workflow_id": payload.workflow_id,
        "document_id": payload.document_id,
        "content_hash": payload.content_hash,
        "external_id": payload.external_id,
        "provider": payload.provider,
        "canonical_source": payload.canonical_source,
        "origin_uri": payload.origin_uri,
        "source": payload.source,
        "media_type": payload.media_type,
        "title": payload.title,
        "language": payload.language,
        "tags": list(payload.tags),
        "parser_stats": dict(payload.parser_stats),
        "provider_tags": dict(payload.provider_tags),
    }
    return len(json.dumps(payload_dict, sort_keys=True).encode("utf-8"))


def _policy_action_from_ingestion(status: IngestionStatus) -> str:
    if status is IngestionStatus.UPSERT:
        return "upsert"
    if status is IngestionStatus.SKIP:
        return "skip"
    return "flag"
