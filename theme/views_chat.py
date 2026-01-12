from __future__ import annotations

from uuid import uuid4

from django.shortcuts import render
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

from ai_core.graph.core import GraphContext, ThreadAwareCheckpointer
from common.constants import (
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_THREAD_ID_HEADER,
)
from theme.chat_utils import (
    append_history,
    build_hybrid_config,
    build_snippet_items,
    coerce_optional_text,
    load_history,
    resolve_history_limit,
    trim_history,
)

logger = get_logger(__name__)
CHECKPOINTER = ThreadAwareCheckpointer()


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
            run as run_rag_graph,
        )

        views = _views()
        scope = views._scope_context_from_request(request)
        tenant_id = scope.tenant_id
        tenant_schema = scope.tenant_schema or tenant_id

        # 1. Prepare State & Meta
        state = {
            "schema_id": RAG_SCHEMA_ID,
            "schema_version": RAG_IO_VERSION_STRING,
            "question": message,
            "query": message,  # Required for retrieval
            "hybrid": build_hybrid_config(request),
        }

        from ai_core.contracts.business import BusinessContext

        business = BusinessContext(
            case_id=case_id,
            collection_id=collection_id,
            workflow_id="rag-chat-manual",
            thread_id=thread_id,
        )
        tool_context = scope.to_tool_context(
            business=business,
            metadata={"graph_name": "rag.default", "graph_version": "v0"},
        )
        graph_context = GraphContext(
            tool_context=tool_context,
            graph_name="rag.default",
            graph_version="v0",
        )

        meta = {
            "scope_context": scope.model_dump(mode="json", exclude_none=True),
            "business_context": business.model_dump(mode="json", exclude_none=True),
            # Ensure we have a valid tool context
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

        history_limit = resolve_history_limit()
        history = []
        try:
            history = load_history(CHECKPOINTER.load(graph_context))
        except Exception:
            logger.exception(
                "chat_submit.checkpoint_load_failed",
                extra={"thread_id": thread_id},
            )

        state["chat_history"] = list(history)

        # 2. Run Graph
        final_state, result_payload = run_rag_graph(state, meta)

        # 3. Extract Answer
        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])
        snippet_items = build_snippet_items(snippets)

        append_history(history, role="user", content=message)
        append_history(history, role="assistant", content=answer)
        history = trim_history(history, limit=history_limit)
        try:
            CHECKPOINTER.save(graph_context, {"chat_history": history})
        except Exception:
            logger.exception(
                "chat_submit.checkpoint_save_failed",
                extra={"thread_id": thread_id},
            )

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
