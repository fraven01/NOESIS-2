from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from ai_core.graphs import crawler_ingestion_graph
from crawler.contracts import NormalizedSource, normalize_source
from crawler.fetcher import FetchRequest, PolitenessContext
from crawler.frontier import CrawlSignals, FrontierAction, SourceDescriptor
from crawler.guardrails import GuardrailLimits, GuardrailSignals, GuardrailStatus
from crawler.ingestion import IngestionStatus
from crawler.parser import ParseStatus, ParserContent, ParserStats


def _build_state(
    url: str = "https://example.com/doc",
) -> Tuple[Dict[str, object], Dict[str, str], bytes]:
    source: NormalizedSource = normalize_source("web", url, None)
    descriptor = SourceDescriptor(host="example.com", path="/doc")
    frontier_input = {
        "descriptor": descriptor,
        "signals": CrawlSignals(),
    }
    politeness = PolitenessContext(host="example.com")
    request = FetchRequest(
        canonical_source=source.canonical_source, politeness=politeness
    )
    body = b"<html><body>Example content for crawler.</body></html>"
    fetch_input = {
        "request": request,
        "status_code": 200,
        "body": body,
        "headers": {"Content-Type": "text/html"},
        "elapsed": 0.12,
    }
    parse_content = ParserContent(
        media_type="text/html",
        primary_text="Example content for crawler.",
        title="Example",
        content_language="en",
    )
    parse_stats = ParserStats(
        token_count=5, character_count=32, extraction_path="html.body"
    )
    parse_input = {
        "status": ParseStatus.PARSED,
        "content": parse_content,
        "stats": parse_stats,
    }
    guardrail_signals = GuardrailSignals(
        tenant_id="tenant",
        provider=source.provider,
        canonical_source=source.canonical_source,
        host="example.com",
        document_bytes=len(body),
    )
    gating_input = {
        "limits": GuardrailLimits(),
        "signals": guardrail_signals,
    }
    state: Dict[str, object] = {
        "tenant_id": "tenant",
        "case_id": "case",
        "workflow_id": "workflow",
        "external_id": source.external_id,
        "origin_uri": source.canonical_source,
        "provider": source.provider,
        "frontier_input": frontier_input,
        "fetch_input": fetch_input,
        "parse_input": parse_input,
        "normalize_input": {
            "source": source,
            "document_id": "doc-1",
            "tags": ("alpha",),
        },
        "delta_input": {},
        "gating_input": gating_input,
    }
    meta = {"tenant_id": "tenant", "case_id": "case", "workflow_id": "workflow"}
    return state, meta, body


def test_nominal_run_executes_pipeline() -> None:
    initial_state, meta, body = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()

    assert "transitions" not in initial_state

    state = graph.start_crawl(initial_state)
    result_state, result = graph.run(state, meta)

    assert initial_state is not state
    assert result_state["content_hash"] is not None
    assert pytest.approx(result_state["gating_score"]) == 1.0
    assert isinstance(result["graph_run_id"], str) and result["graph_run_id"]
    transitions = result_state["transitions"]
    assert transitions["crawler.frontier"]["decision"] == FrontierAction.ENQUEUE.value
    assert transitions["crawler.fetch"]["decision"] == "fetched"
    assert transitions["crawler.parse"]["decision"] == ParseStatus.PARSED.value
    assert transitions["crawler.normalize"]["decision"] == "normalized"
    assert transitions["crawler.delta"]["decision"] in {"new", "changed"}
    assert (
        transitions["crawler.ingest_decision"]["decision"]
        == IngestionStatus.UPSERT.value
    )
    assert transitions["crawler.store"]["decision"] == "stored"
    assert transitions["rag.upsert"]["decision"] == "upsert"
    assert transitions["rag.retire"]["decision"] == "skip"
    assert result["decision"] == IngestionStatus.UPSERT.value
    assert result["attributes"].get("severity") == "info"
    assert result_state["origin_uri"] == initial_state["origin_uri"]
    # Ensure original input dictionary remains untouched.
    assert "transitions" not in initial_state


def test_start_crawl_respects_legacy_manual_review_control() -> None:
    initial_state, _, _ = _build_state()
    initial_state["control"] = {"manual_review": "required", "dry_run": True}
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()

    state = graph.start_crawl(initial_state)

    assert state["control"]["review"] == "required"
    assert state["control"]["manual_review"] == "required"
    assert state["control"]["dry_run"] is True


def test_manual_approval_path() -> None:
    initial_state, meta, body = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)
    # Force guardrail deny by shrinking limit below document size.
    limits = GuardrailLimits(max_document_bytes=10)
    state["gating_input"]["limits"] = limits

    denied_state, denied_result = graph.run(state, meta)
    transitions = denied_state["transitions"]
    gating = transitions["crawler.gating"]
    assert gating["decision"] == GuardrailStatus.DENY.value
    assert denied_state["control"]["review"] == "required"
    assert denied_state["control"]["manual_review"] == "required"
    assert gating["attributes"]["severity"] == "warn"
    assert "crawler.ingest_decision" not in transitions
    assert denied_result["decision"] == "pending"
    assert denied_result["graph_run_id"]

    approved = graph.approve_ingest(denied_state)
    approved_state, final_result = graph.run(approved, meta)
    assert approved_state["control"].get("review") is None
    assert approved_state["control"].get("manual_review") is None
    assert (
        approved_state["transitions"]["crawler.gating"]["decision"]
        == GuardrailStatus.ALLOW.value
    )
    assert approved_state["transitions"]["crawler.store"]["decision"] == "stored"
    assert approved_state["transitions"]["rag.upsert"]["decision"] == "upsert"
    assert final_result["decision"] == IngestionStatus.UPSERT.value
    assert final_result["graph_run_id"] != denied_result["graph_run_id"]


def test_retire_flow_dispatches_retire_node() -> None:
    initial_state, meta, _ = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)
    state = graph.retire(state)

    retired_state, result = graph.run(state, meta)
    transitions = retired_state["transitions"]
    assert (
        transitions["crawler.ingest_decision"]["decision"]
        == IngestionStatus.RETIRE.value
    )
    assert transitions["crawler.store"]["decision"] == "skip"
    assert transitions["rag.retire"]["decision"] == "retire"
    assert result["decision"] == IngestionStatus.RETIRE.value


def test_shadow_mode_turns_upsert_into_noop() -> None:
    initial_state, meta, _ = _build_state()
    recorded = {"calls": 0}

    def _recording_upsert(decision):
        recorded["calls"] += 1
        return {
            "status": "queued",
            "document": decision.payload.document_id if decision.payload else None,
        }

    graph = crawler_ingestion_graph.CrawlerIngestionGraph(
        upsert_handler=_recording_upsert
    )
    state = graph.start_crawl(initial_state)
    state = graph.shadow_mode_on(state)

    shadow_state, result = graph.run(state, meta)
    assert recorded["calls"] == 0
    transitions = shadow_state["transitions"]
    assert transitions["rag.upsert"]["decision"] == "shadow_skip"
    assert result["decision"] == IngestionStatus.UPSERT.value


def test_shadow_mode_toggle_allows_follow_up_upsert() -> None:
    initial_state, meta, _ = _build_state()
    recorded: List[int] = []

    def _recording_upsert(decision):
        recorded.append(decision.payload.document_id if decision.payload else None)
        return {"status": "queued"}

    graph = crawler_ingestion_graph.CrawlerIngestionGraph(
        upsert_handler=_recording_upsert
    )
    state = graph.start_crawl(initial_state)
    state = graph.shadow_mode_on(state)

    shadow_state, first_result = graph.run(state, meta)
    assert recorded == []
    assert shadow_state["transitions"]["rag.upsert"]["decision"] == "shadow_skip"

    resumed_state = graph.shadow_mode_off(shadow_state)
    resumed_state, resumed_result = graph.run(resumed_state, meta)
    assert recorded != []
    assert resumed_state["transitions"]["rag.upsert"]["decision"] == "upsert"
    assert resumed_result["graph_run_id"] != first_result["graph_run_id"]


def test_event_emitter_receives_all_nodes() -> None:
    initial_state, meta, _ = _build_state()
    events: List[Tuple[str, str, str]] = []

    def _emit(node: str, transition, run_id: str) -> None:
        events.append((node, transition.decision, run_id))

    graph = crawler_ingestion_graph.CrawlerIngestionGraph(event_emitter=_emit)
    state = graph.start_crawl(initial_state)
    _, result = graph.run(state, meta)

    assert {event[0] for event in events} == {
        "crawler.frontier",
        "crawler.fetch",
        "crawler.parse",
        "crawler.normalize",
        "crawler.delta",
        "crawler.gating",
        "crawler.ingest_decision",
        "crawler.store",
        "rag.upsert",
        "rag.retire",
    }
    assert all(event[2] == result["graph_run_id"] for event in events)


def test_missing_required_keys_raise() -> None:
    state, meta, _ = _build_state()
    state.pop("tenant_id")
    meta = {"case_id": meta["case_id"], "workflow_id": meta["workflow_id"]}
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    prepared = graph.start_crawl(state)

    with pytest.raises(KeyError) as exc:
        graph.run(prepared, meta)

    assert exc.value.args[0] == "missing_required_state_keys"


def test_ingestion_missing_delta_emits_missing_artifact() -> None:
    initial_state, meta, _ = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    running_state = graph.start_crawl(initial_state)
    complete_state, _ = graph.run(running_state, meta)

    artifacts = dict(complete_state["artifacts"])
    artifacts.pop("delta_decision")
    control = dict(complete_state["control"])
    state_copy = dict(complete_state)

    transition, should_continue = graph._run_ingestion(state_copy, artifacts, control)

    assert transition.decision == "missing_artifact"
    assert transition.attributes["artifact"] == "delta_decision"
    assert transition.attributes["severity"] == "error"
    assert should_continue is False
    assert state_copy["ingest_action"] == "error"


def test_retire_overrides_recompute_delta() -> None:
    initial_state, meta, _ = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)
    state = graph.recompute_delta(state)
    state = graph.retire(state)

    final_state, result = graph.run(state, meta)
    ingestion_transition = final_state["transitions"]["crawler.ingest_decision"]
    assert ingestion_transition["decision"] == IngestionStatus.RETIRE.value
    assert (
        ingestion_transition["attributes"].get("conflict_resolution")
        == "retire_overrides_recompute"
    )
    assert final_state["control"]["recompute_delta"] is False
    assert result["decision"] == IngestionStatus.RETIRE.value


def test_gating_score_uses_custom_score_if_supplied() -> None:
    initial_state, meta, _ = _build_state()
    initial_state["gating_input"]["score"] = 0.42
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)

    result_state, result = graph.run(state, meta)
    assert pytest.approx(result_state["gating_score"]) == pytest.approx(0.42)
    assert result["attributes"].get("severity") == "info"


def test_parser_failure_sets_error_summary() -> None:
    initial_state, meta, _ = _build_state()
    initial_state["parse_input"].update(
        {"status": ParseStatus.PARSER_FAILURE, "content": None, "stats": None}
    )
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)

    _, result = graph.run(state, meta)
    assert result["decision"] == "skip"
    assert result["reason"] == ParseStatus.PARSER_FAILURE.value
    assert result["attributes"].get("severity") == "error"


def test_store_only_mode_stores_without_upsert() -> None:
    initial_state, meta, _ = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)
    state["control"]["mode"] = "store_only"

    result_state, _ = graph.run(state, meta)
    transitions = result_state["transitions"]
    assert transitions["crawler.store"]["decision"] == "stored"
    assert transitions["rag.upsert"]["decision"] == "skip"
    assert transitions["rag.upsert"]["reason"] == "mode_disabled"


def test_dry_run_blocks_store_and_upsert() -> None:
    initial_state, meta, _ = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)
    state["control"]["dry_run"] = True

    result_state, _ = graph.run(state, meta)
    transitions = result_state["transitions"]
    assert transitions["crawler.store"]["decision"] == "skip"
    assert transitions["crawler.store"]["reason"] == "dry_run"
    assert transitions["rag.upsert"]["decision"] == "skip"
    assert transitions["rag.upsert"]["reason"] == "dry_run"
    ingestion_attrs = result_state["transitions"]["crawler.ingest_decision"][
        "attributes"
    ]
    assert ingestion_attrs["blocked"] == "dry_run"


def test_review_required_blocks_store_and_upsert() -> None:
    initial_state, meta, _ = _build_state()
    graph = crawler_ingestion_graph.CrawlerIngestionGraph()
    state = graph.start_crawl(initial_state)
    state["control"]["review"] = "required"

    result_state, _ = graph.run(state, meta)
    transitions = result_state["transitions"]
    assert transitions["crawler.gating"]["attributes"]["review"] == "required"
    assert transitions["crawler.store"]["decision"] == "skip"
    assert transitions["crawler.store"]["reason"] == "review_required"
    assert transitions["rag.upsert"]["decision"] == "skip"
    assert transitions["rag.upsert"]["reason"] == "review_required"
    ingestion_attrs = result_state["transitions"]["crawler.ingest_decision"][
        "attributes"
    ]
    assert ingestion_attrs["blocked"] == "review_required"
