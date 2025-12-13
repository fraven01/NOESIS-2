"""External Knowledge Graph using LangGraph."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import HttpUrl

from ai_core.infra.observability import observe_span
from ai_core.tools.web_search import (
    SearchProviderError,
    WebSearchResponse,
    WebSearchWorker,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- Protocols
class IngestionTrigger(Protocol):
    """Protocol for triggering document ingestion."""

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """Trigger the ingestion pipeline and return result metadata."""


# --------------------------------------------------------------------- State
class ExternalKnowledgeState(TypedDict):
    """Runtime state for the external knowledge graph."""

    # Input
    query: str
    collection_id: str
    enable_hitl: bool
    context: dict[str, Any]  # Tenant ID, Trace ID, etc.

    # Output / Intermediate
    search_results: list[dict[str, Any]]
    selected_result: dict[str, Any] | None
    ingestion_result: dict[str, Any] | None
    error: str | None
    auto_ingest: bool


class GraphConfig(TypedDict):
    """Configuration passed via configurable."""

    search_worker: WebSearchWorker
    ingestion_trigger: IngestionTrigger | None
    top_n: int
    min_snippet_length: int
    blocked_domains: list[str]
    prefer_pdf: bool


# --------------------------------------------------------------------- Nodes
def _normalise_url(url: str) -> str:
    return str(HttpUrl(url))


def _blocked_domain(url: str, blocked_domains: list[str]) -> bool:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if not hostname or not blocked_domains:
        return False
    for domain in blocked_domains:
        blocked = domain.lower()
        if hostname == blocked or hostname.endswith(f".{blocked}"):
            return True
    return False


@observe_span(name="node.search")
def search_node(state: ExternalKnowledgeState) -> dict[str, Any]:
    """Execute web search using the configured worker."""
    query = state["query"]
    context = state.get("context", {})
    worker = context.get("runtime_worker")

    if not worker:
        return {"error": "No search worker configured in context"}

    # Filter context to strictly allowed fields for WebSearchWorker
    telemetry_ctx = {
        k: v
        for k, v in context.items()
        if k
        in {
            "tenant_id",
            "trace_id",
            "workflow_id",
            "case_id",
            "run_id",
            "worker_call_id",
        }
    }

    try:
        response: WebSearchResponse = worker.run(query=query, context=telemetry_ctx)
    except SearchProviderError as exc:
        logger.warning(f"Search failed: {exc}")
        return {"error": str(exc), "search_results": []}

    results = [r.model_dump(mode="json") for r in response.results]
    return {"search_results": results, "error": None}


@observe_span(name="node.select")
def selection_node(state: ExternalKnowledgeState) -> dict[str, Any]:
    """Filter and select the best candidate from search results."""
    results = state.get("search_results", [])
    context = state.get("context", {})
    # Config values like min_len could still come from a config object if needed,
    # but for consistency/simplicity we'll expect them in context or use defaults
    min_len = context.get("min_snippet_length", 40)
    blocked = context.get("blocked_domains", [])
    top_n = context.get("top_n", 5)
    prefer_pdf = context.get("prefer_pdf", True)

    validated = []
    for raw in results:
        url = raw.get("url")
        snippet = raw.get("snippet", "")
        if not url or len(snippet) < min_len:
            continue
        if _blocked_domain(url, blocked):
            continue

        # Check noindex (simple heuristic)
        lowered = snippet.lower()
        if "noindex" in lowered and "robot" in lowered:
            continue

        validated.append(raw)

    shortlisted = validated[:top_n]
    selected = None

    if shortlisted:
        if prefer_pdf:
            for cand in shortlisted:
                if cand.get("is_pdf"):
                    selected = cand
                    break

        if not selected:
            # Fallback to highest score (assuming list is sorted by relevance from search)
            # or purely first item
            selected = shortlisted[0]

    return {"selected_result": selected}


@observe_span(name="node.ingest")
def ingestion_node(state: ExternalKnowledgeState) -> dict[str, Any]:
    """Trigger ingestion for the selected result."""
    selected = state.get("selected_result")
    context = state.get("context", {})
    if not selected:
        return {"ingestion_result": {"status": "skipped", "reason": "no_selection"}}

    trigger = context.get("runtime_trigger")
    if not trigger:
        return {
            "ingestion_result": {"status": "error", "reason": "no_trigger_configured"}
        }

    url = selected.get("url")
    collection_id = state.get("collection_id")

    # Filter context (though implementation might not be strict, better safe)
    telemetry_ctx = {
        k: v
        for k, v in context.items()
        if k
        in {
            "tenant_id",
            "trace_id",
            "workflow_id",
            "case_id",
            "run_id",
            "worker_call_id",
        }
    }

    try:
        result = trigger.trigger(
            url=url,
            collection_id=collection_id,
            context=telemetry_ctx,
        )
        return {"ingestion_result": result}
    except Exception as exc:
        logger.exception("Ingestion trigger failed")
        return {"ingestion_result": {"status": "error", "reason": str(exc)}}


# --------------------------------------------------------------------- Graph Definition

workflow = StateGraph(ExternalKnowledgeState)

workflow.add_node("search", search_node)
workflow.add_node("select", selection_node)
workflow.add_node("ingest", ingestion_node)

workflow.add_edge(START, "search")
workflow.add_edge("search", "select")


def _check_hitl(state: ExternalKnowledgeState) -> Literal["ingest", "end"]:
    # Simplification: If HITL is enabled, we might pause here.
    # But for now, user requested "functionality similar to present".
    # The previous graph had HITL.
    # To properly support HITL in LangGraph, we would use an interrupt.
    # However, to keep it simple for this refactor step (and since user said "input -> output"),
    # we will implement the automatic flow first.
    # If hitl is true, we might just skip ingest or return early?
    # User said: "actually we simple need an input... graph output... business logic uses it".

    # Logic: If nothing selected, end.
    if not state.get("selected_result"):
        return "end"

    # If auto_ingest is explicitly False (default), stop here.
    # We only proceed to ingest if auto_ingest is True.
    if not state.get("auto_ingest", False):
        return "end"

    return "ingest"


workflow.add_conditional_edges("select", _check_hitl, {"ingest": "ingest", "end": END})
workflow.add_edge("ingest", END)

# Compiled Graph
graph = workflow.compile()
