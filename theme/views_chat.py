from __future__ import annotations

from django.http import HttpResponse
from django.views.decorators.http import require_POST
from structlog.stdlib import get_logger

logger = get_logger(__name__)


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
    case_id = (
        request.POST.get("case_id")
        or request.headers.get("X-Case-ID")
        or "dev-case-local"
    )

    # Feature: Global Search in RAG Chat (Dev Workbench)
    # If global_search is checked, ignore case_id to search entire tenant
    if request.POST.get("global_search") == "on":
        case_id = None

    if not message:
        return HttpResponse('<div class="text-red-500 p-4">Message is required.</div>')

    try:
        from ai_core.graphs.technical.retrieval_augmented_generation import (
            run as run_rag_graph,
        )

        views = _views()
        tenant_id, tenant_schema = views._tenant_context_from_request(request)

        # 1. Prepare State & Meta
        state = {
            "question": message,
            "query": message,  # Required for retrieval
            "hybrid": {
                "alpha": 0.5,
                "top_k": 5,
                "min_sim": 0.0,
            },
        }

        from ai_core.contracts.business import BusinessContext
        from ai_core.ids.http_scope import normalize_request

        scope = normalize_request(request)
        business = BusinessContext(
            case_id=case_id,
            workflow_id="rag-chat-manual",
        )
        tool_context = scope.to_tool_context(
            business=business,
            metadata={"graph_name": "rag.default", "graph_version": "v0"},
        )

        meta = {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema or scope.tenant_schema,
            "case_id": case_id,
            "trace_id": scope.trace_id,
            "run_id": scope.run_id,  # Required for ScopeContext validation
            "workflow_id": "rag-chat-manual",  # Workflow type for tracing
            "scope_context": scope.model_dump(mode="json", exclude_none=True),
            "business_context": business.model_dump(mode="json", exclude_none=True),
            # Ensure we have a valid tool context
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

        # 2. Run Graph
        final_state, result_payload = run_rag_graph(state, meta)

        # 3. Extract Answer
        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])

        # 4. Render Response Partial
        # We'll inline the HTML for now, or we could create a partial template
        response_html = f"""
        <div class="flex items-start gap-4 justify-end">
            <div class="bg-indigo-600 p-4 rounded-2xl rounded-tr-none shadow-sm text-sm text-white max-w-[80%]">
                <p>{message}</p>
            </div>
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-600 font-bold text-xs">You</div>
        </div>
        <div class="flex items-start gap-4">
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold text-xs">AI</div>
            <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-slate-100 text-sm text-slate-700 max-w-[80%] space-y-2">
                <div class="prose prose-sm max-w-none text-slate-700">
                    {answer}
                </div>
                {'<div class="mt-2 pt-2 border-t border-slate-100"><p class="text-xs font-semibold text-slate-500 mb-1">Sources:</p><ul class="space-y-1">' + ''.join([f'<li class="text-xs text-slate-400 truncate" title="{s.get("text", "")[:200]}">• {s.get("source", "Unknown")} ({int(s.get("score", 0)*100)}%)</li>' for s in snippets[:3]]) + '</ul></div>' if snippets else ''}
            </div>
        </div>
        """
        return HttpResponse(response_html)

    except Exception as e:
        logger.exception("chat_submit.failed")
        return HttpResponse(
            f"""
        <div class="flex items-start gap-4">
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-red-100 flex items-center justify-center text-red-700 font-bold text-xs">ERR</div>
            <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-red-100 text-sm text-red-600 max-w-[80%]">
                <p>Error processing request: {str(e)}</p>
            </div>
        </div>
        """
        )
