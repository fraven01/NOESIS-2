from __future__ import annotations

from uuid import uuid4

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

# from ai_core.graph.core import GraphContext, ThreadAwareCheckpointer (Removed M-5)
from ai_core.services.rag_query import RagQueryService
from common.constants import (
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_THREAD_ID_HEADER,
)
from theme.chat_utils import (
    build_hybrid_config,
    build_snippet_items,
    coerce_optional_text,
    link_citations,
)

logger = get_logger(__name__)
logger = get_logger(__name__)
# CHECKPOINTER = ThreadAwareCheckpointer() (Removed M-5)


def _views():
    from theme import views as theme_views

    return theme_views


def _build_used_source_items(
    used_sources: object,
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    if not isinstance(used_sources, list):
        return []
    if limit is not None and limit > 0:
        sources = used_sources[:limit]
    else:
        sources = used_sources
    items: list[dict[str, object]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        label = source.get("label") or source.get("id") or "Source"
        try:
            relevance = float(source.get("relevance_score", 0))
        except (TypeError, ValueError):
            relevance = 0.0
        items.append(
            {
                "label": str(label),
                "score_percent": max(0, min(100, int(relevance * 100))),
                "id": source.get("id"),
            }
        )
    return items


@require_POST
def chat_submit(request):
    """
    Handle HTMX submission for RAG Chat.
    Invokes the production RAG graph and returns the assistant's reply.
    """
    message = request.POST.get("message")
    case_id = coerce_optional_text(
        request.POST.get("case_id") or request.headers.get(X_CASE_ID_HEADER)
    )
    # No dev-case-local fallback (SCOPE-1): Scope logic below will set to None if needed
    collection_id = coerce_optional_text(
        request.POST.get("collection_id") or request.headers.get(X_COLLECTION_ID_HEADER)
    )
    thread_id = coerce_optional_text(
        request.POST.get("thread_id") or request.headers.get(X_THREAD_ID_HEADER)
    )
    if thread_id is None:
        thread_id = uuid4().hex
    try:
        request.session["rag_chat_thread_id"] = thread_id
    except Exception:
        pass

    scope_option = (
        request.POST.get("chat_scope")
        or request.session.get("rag_chat_scope")
        or "collection"
    )
    if request.POST.get("global_search") == "on":
        scope_option = "global"
    if scope_option not in {"collection", "case", "global"}:
        scope_option = "collection"
    request.session["rag_chat_scope"] = scope_option

    manual_collection_id = None

    needs_manual_collection = False
    if scope_option == "collection":
        case_id = None
    elif scope_option == "case":
        collection_id = None
    else:  # global
        case_id = None
        collection_id = None
        needs_manual_collection = True

    if not message:
        return render(
            request,
            "theme/partials/chat_message.html",
            {"error": "Message is required."},
        )

    try:
        from theme.helpers.context import prepare_workbench_context

        # 1. Prepare Context
        tool_context = prepare_workbench_context(
            request,
            case_id=case_id,
            collection_id=collection_id,
            thread_id=thread_id,
            workflow_id="rag-chat-manual",
        )

        # Add graph metadata manually (helper doesn't add it)
        tool_context = tool_context.model_copy(
            update={"metadata": {"graph_name": "rag.default", "graph_version": "v0"}}
        )

        # graph_context removed (M-5)

        scope = tool_context.scope
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id
        views = _views()
        manual_collection_id, _ = views._resolve_manual_collection(tenant_id, None)
        if needs_manual_collection and manual_collection_id:
            collection_id = manual_collection_id

        service = RagQueryService()
        _, result_payload = service.execute(
            tool_context=tool_context,
            question=message,
            hybrid=build_hybrid_config(request),
        )

        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])
        retrieval_meta = result_payload.get("retrieval") or {}
        try:
            top_k = int(retrieval_meta.get("top_k_effective") or 0)
        except (TypeError, ValueError):
            top_k = 0
        snippet_limit = top_k or len(snippets) or None
        snippet_items = build_snippet_items(snippets, limit=snippet_limit)
        answer = link_citations(answer, snippet_items)
        reasoning = result_payload.get("reasoning")
        if not isinstance(reasoning, dict):
            reasoning = None
        used_sources = _build_used_source_items(
            result_payload.get("used_sources"),
            limit=snippet_limit,
        )
        suggested_followups = result_payload.get("suggested_followups")
        if not isinstance(suggested_followups, list):
            suggested_followups = []
        debug_meta = result_payload.get("debug_meta")
        show_debug = bool(settings.DEBUG) or bool(
            getattr(getattr(request, "user", None), "is_staff", False)
        )
        if not show_debug:
            debug_meta = None

        return render(
            request,
            "theme/partials/chat_message.html",
            {
                "message": message,
                "answer": answer,
                "snippets": snippet_items,
                "reasoning": reasoning,
                "used_sources": used_sources,
                "suggested_followups": suggested_followups,
                "debug_meta": debug_meta,
                "show_debug": show_debug,
                "tenant_id": tenant_id,
                "tenant_schema": tenant_schema or scope.tenant_schema,
                "case_id": case_id,
                "collection_id": collection_id,
                "thread_id": thread_id,
                "chat_scope": scope_option,
            },
        )

    except Exception as e:
        logger.exception("chat_submit.failed")
        return render(
            request,
            "theme/partials/chat_message.html",
            {"error": f"Error processing request: {str(e)}"},
        )
