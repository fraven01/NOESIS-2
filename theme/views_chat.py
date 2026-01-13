from __future__ import annotations

from uuid import uuid4

from django.shortcuts import render
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

# from ai_core.graph.core import GraphContext, ThreadAwareCheckpointer (Removed M-5)
from common.constants import (
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_THREAD_ID_HEADER,
)
from theme.chat_utils import (
    build_hybrid_config,
    build_snippet_items,
    coerce_optional_text,
)

logger = get_logger(__name__)
logger = get_logger(__name__)
# CHECKPOINTER = ThreadAwareCheckpointer() (Removed M-5)


def _views():
    from theme import views as theme_views

    return theme_views


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
    if case_id is None:
        case_id = "dev-case-local"
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

    # Feature: Global Search in RAG Chat (Dev Workbench)
    # If global_search is checked, ignore case_id to search entire tenant
    if request.POST.get("global_search") == "on":
        case_id = None

    if not message:
        return render(
            request,
            "theme/partials/chat_message.html",
            {"error": "Message is required."},
        )

    try:
        from ai_core.graphs.technical.retrieval_augmented_generation import (
            RAG_IO_VERSION_STRING,
            RAG_SCHEMA_ID,
        )
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

        # 2. Prepare State & Meta
        state = {
            "schema_id": RAG_SCHEMA_ID,
            "schema_version": RAG_IO_VERSION_STRING,
            "question": message,
            "query": message,  # Required for retrieval
            "hybrid": build_hybrid_config(request),
        }

        # History loading removed (M-5), handled by Graph

        from theme.helpers.tasks import submit_business_graph

        # 3. Run Graph via Worker (M-2)
        # We wait for the result to maintain the synchronous UI experience for now
        response_payload, completed = submit_business_graph(
            graph_name="rag.default",
            tool_context=tool_context,
            state=state,
            timeout_s=30,
        )

        if not completed:
            logger.warning("chat_submit.timeout", extra={"thread_id": thread_id})
            return render(
                request,
                "theme/partials/chat_message.html",
                {"error": "Request timed out. Please try again."},
            )

        # Parse generic worker response
        if response_payload.get("status") == "error":
            logger.error(
                "chat_submit.worker_error", error=response_payload.get("error")
            )
            return render(
                request,
                "theme/partials/chat_message.html",
                {"error": "An error occurred during processing."},
            )

        data = response_payload.get("data", {})
        # Note: final_state might not be returned depending on graph, but usually is
        # result_payload usually contains "answer"
        result_payload = data.get("result", {})

        # 4. Extract Answer
        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])
        snippet_items = build_snippet_items(snippets)

        # History saving removed (M-5), handled by Graph

        # 4. Render Response Partial
        return render(
            request,
            "theme/partials/chat_message.html",
            {
                "message": message,
                "answer": answer,
                "snippets": snippet_items,
                "tenant_id": tenant_id,
                "tenant_schema": tenant_schema or scope.tenant_schema,
                "case_id": case_id,
                "collection_id": collection_id,
                "thread_id": thread_id,
            },
        )

    except Exception as e:
        logger.exception("chat_submit.failed")
        return render(
            request,
            "theme/partials/chat_message.html",
            {"error": f"Error processing request: {str(e)}"},
        )
