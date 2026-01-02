"""Web Acquisition Graph for External Knowledge Search.

Rules:
1. Strict ToolContext usage.
2. Worker injection via 'configurable' config.
3. Output: SearchResults or NormalizedDocument.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ai_core.infra.observability import observe_span
from ai_core.tools.shared_workers import get_web_search_worker
from ai_core.tool_contracts import ToolContext
from ai_core.tools.web_search import (
    SearchProviderError,
    WebSearchResponse,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- State
class WebAcquisitionInput(TypedDict):
    """Input payload for the web acquisition graph."""

    query: str
    search_config: dict[str, Any] | None
    # Optional preselected results to bypass search (e.g. from UI)
    preselected_results: list[dict[str, Any]] | None
    # Mode: 'search_only' (default) or 'select_best'
    mode: Literal["search_only", "select_best"] | None


class WebAcquisitionOutput(TypedDict):
    """Output contract for the web acquisition graph."""

    search_results: list[dict[str, Any]]
    selected_result: dict[str, Any] | None
    decision: Literal["acquired", "error", "no_results"]
    error: str | None
    telemetry: dict[str, Any]


class WebAcquisitionState(TypedDict):
    """Runtime state for the web acquisition graph."""

    input: WebAcquisitionInput
    # ToolContext is validated once and stored here
    tool_context: ToolContext | None

    # Internal state
    search_results: list[dict[str, Any]]
    selected_result: dict[str, Any] | None

    error: str | None
    output: WebAcquisitionOutput | None


# --------------------------------------------------------------------- Helpers


def _run_search_with_timeout(worker, query: str, context: ToolContext, timeout: int):
    """Run search worker with timeout protection."""
    result_queue: Queue = Queue()

    def worker_thread():
        try:
            response = worker.run(query=query, context=context)
            result_queue.put(("success", response))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"Search worker exceeded {timeout}s timeout")

    try:
        status, result = result_queue.get_nowait()
        if status == "error":
            raise result
        return result
    except Empty:
        raise TimeoutError("Search worker produced no result")


def _check_search_rate_limit(tenant_id: str, query: str) -> bool:
    """Check if tenant has exceeded search rate limit."""
    from django.conf import settings
    from django.core.cache import cache

    cache_key = f"search_rate_limit:{tenant_id}"
    count = cache.get(cache_key, 0)

    max_searches_per_hour = getattr(settings, "MAX_SEARCHES_PER_TENANT_PER_HOUR", 100)

    if count >= max_searches_per_hour:
        logger.warning(
            "Search rate limit exceeded",
            extra={
                "tenant_id": tenant_id,
                "query": query[:100],
                "count": count,
                "limit": max_searches_per_hour,
            },
        )
        return False

    cache.set(cache_key, count + 1, timeout=3600)
    return True


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


# --------------------------------------------------------------------- Nodes


@observe_span(name="node.validate_input")
def validate_input_node(
    state: WebAcquisitionState, config: RunnableConfig
) -> dict[str, Any]:
    """Validate input and initialize ToolContext."""
    inp = state.get("input", {})
    # Note: config.configurable can be used for future extensions

    # 1. Recover Context from configurable (passed by invocation)
    # Ideally LangGraph invocations pass 'context' in configurable
    # For now, we assume caller passes it in the state "context" key if not in configurable
    # BUT architectural fix says: "tool_context will be validated once and kept in state"
    # So we expect the caller to pass 'tool_context' in state OR a raw dict 'context' we parse.

    # Let's support both for transition:
    # 1. 'tool_context' already in state (ideal)
    # 2. 'context' dict in state (legacy adapter)

    tool_context = state.get("tool_context")
    if not tool_context:
        raw_context = state.get("context")  # Legacy hook
        if raw_context:
            try:
                tool_context = ToolContext.model_validate(raw_context)
            except ValidationError as ve:
                return {"error": f"Invalid context: {ve}"}
        else:
            return {"error": "Missing 'tool_context' or 'context' in input state"}

    if not inp.get("query") and not inp.get("preselected_results"):
        return {"error": "Missing 'query' or 'preselected_results'"}

    return {"tool_context": tool_context, "error": None}


@observe_span(name="node.search")
def search_node(state: WebAcquisitionState, config: RunnableConfig) -> dict[str, Any]:
    """Execute search."""
    # Check upstream error
    if state.get("error"):
        return {}

    inp = state.get("input", {})
    tool_context = state.get("tool_context")
    if not tool_context:
        return {"error": "Context lost"}

    preselected = inp.get("preselected_results")
    if preselected:
        return {"search_results": preselected}

    query = inp.get("query")
    if not query:
        return {"error": "Query missing"}

    # Rate Limit
    tenant_id = tool_context.scope.tenant_id
    if tenant_id:
        if not _check_search_rate_limit(str(tenant_id), query):
            return {
                "error": "Search rate limit exceeded.",
                "search_results": [],
            }

    # Worker Injection via Config (Architectural Fix)
    # config is RunnableConfig (dict-like)
    configurable = config.get("configurable", {}) if config else {}
    worker = configurable.get("search_worker")

    if not worker:
        # Fallback to metadata for transition or default factory
        worker = tool_context.metadata.get("runtime_worker") or get_web_search_worker()

    if not worker:
        return {"error": "No search worker configured"}

    # Timeout
    from django.conf import settings

    search_timeout = getattr(settings, "SEARCH_WORKER_TIMEOUT_SECONDS", 30)

    try:
        response: WebSearchResponse = _run_search_with_timeout(
            worker, query, tool_context, timeout=search_timeout
        )
    except TimeoutError as e:
        logger.warning(str(e), extra={"query": query})
        return {"error": str(e), "search_results": []}
    except SearchProviderError as exc:
        logger.warning(f"Search failed: {exc}")
        return {"error": str(exc), "search_results": []}
    except Exception as exc:
        logger.exception("Unexpected search error")
        return {"error": str(exc), "search_results": []}

    if response.outcome.decision == "error":
        error_msg = (
            response.outcome.meta.get("error", {}).get("message")
            or response.outcome.rationale
        )
        return {"error": error_msg, "search_results": []}

    results = [r.model_dump(mode="json") for r in response.results]
    return {"search_results": results}


@observe_span(name="node.select")
def select_node(state: WebAcquisitionState) -> dict[str, Any]:
    """Filter and select best candidate."""
    if state.get("error"):
        return {}

    results = state.get("search_results", [])
    inp = state.get("input", {})
    config = inp.get("search_config") or {}

    min_len = config.get("min_snippet_length", 40)
    blocked = config.get("blocked_domains", [])
    top_n = config.get("top_n", 5)
    prefer_pdf = config.get("prefer_pdf", True)

    # Bypass snippet checks for preselected URLs
    preselected_urls = {
        item["url"]
        for item in (inp.get("preselected_results") or [])
        if item.get("url")
    }

    validated = []
    for raw in results:
        url = raw.get("url")
        snippet = raw.get("snippet", "")

        if not url:
            continue

        if url not in preselected_urls and len(snippet) < min_len:
            continue

        if _blocked_domain(url, blocked):
            continue

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
            selected = shortlisted[0]

    return {"selected_result": selected, "search_results": shortlisted}


@observe_span(name="node.finalize")
def finalize_node(state: WebAcquisitionState) -> dict[str, Any]:
    """Build final output."""
    error = state.get("error")
    tool_context = state.get("tool_context")

    # Telemetry
    if tool_context:
        telemetry = {
            "trace_id": tool_context.scope.trace_id,
            "tenant_id": tool_context.scope.tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        telemetry = {}

    if error:
        return {
            "output": {
                "decision": "error",
                "error": error,
                "search_results": [],
                "selected_result": None,
                "telemetry": telemetry,
            }
        }

    results = state.get("search_results", [])
    selected = state.get("selected_result")

    decision = "acquired" if results else "no_results"

    return {
        "output": {
            "decision": decision,
            "error": None,
            "search_results": results,
            "selected_result": selected,
            "telemetry": telemetry,
        }
    }


# --------------------------------------------------------------------- Graph


def build_web_acquisition_graph() -> StateGraph:
    """Construct the Web Acquisition Graph."""
    workflow = StateGraph(WebAcquisitionState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("search", search_node)
    workflow.add_node("select", select_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "validate_input")

    def check_error(state: WebAcquisitionState) -> str:
        if state.get("error"):
            return "finalize"
        return "search"

    workflow.add_conditional_edges("validate_input", check_error)

    def check_mode(state: WebAcquisitionState) -> str:
        if state.get("error"):
            return "finalize"
        mode = state.get("input", {}).get("mode", "search_only")
        if mode == "select_best":
            return "select"
        return "finalize"

    workflow.add_conditional_edges("search", check_mode)
    workflow.add_edge("select", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
