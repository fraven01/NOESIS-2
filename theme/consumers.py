from __future__ import annotations

import asyncio
from uuid import uuid4

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.conf import settings
from pydantic import ValidationError
from structlog.stdlib import get_logger

from ai_core.graph.core import GraphContext, ThreadAwareCheckpointer
from ai_core.tool_contracts import ContextError
from ai_core.services.rag_query import RagQueryService
from theme.chat_utils import (
    build_hybrid_config_from_payload,
    build_passage_items_for_workbench,
    build_snippet_items,
    link_citations,
    load_history,
)
from theme.websocket_payloads import RagChatPayload
from theme.helpers.context import prepare_workbench_context


logger = get_logger(__name__)
CHECKPOINTER = ThreadAwareCheckpointer()


class RagChatConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self) -> None:
        await self.accept()

    async def receive_json(self, content, **kwargs) -> None:
        try:
            payload = RagChatPayload.model_validate(content)
        except ValidationError as exc:
            logger.info("rag_chat.invalid_payload", extra={"errors": exc.errors()})
            await self.send_json({"event": "error", "error": "Invalid payload."})
            return

        message = payload.message
        tenant_id = payload.tenant_id
        tenant_schema = payload.tenant_schema or tenant_id
        case_id = payload.case_id  # No dev-case-local fallback (SCOPE-1)
        collection_id = payload.collection_id
        thread_id = payload.thread_id or uuid4().hex
        if payload.global_search:
            case_id = None

        try:
            tool_context = prepare_workbench_context(
                self.scope,
                tenant_id=tenant_id,
                tenant_schema=tenant_schema,
                case_id=case_id,
                collection_id=collection_id,
                workflow_id="rag-chat-manual",
                thread_id=thread_id,
            )
        except ContextError as exc:
            logger.warning(
                "rag_chat.context_forbidden",
                extra={"error": str(exc)},
            )
            await self.send_json({"event": "error", "error": "Forbidden."})
            return

        # Add graph metadata (helper doesn't add this)
        tool_context = tool_context.model_copy(
            update={"metadata": {"graph_name": "rag.default", "graph_version": "v0"}}
        )

        graph_context = GraphContext(
            tool_context=tool_context,
            graph_name="rag.default",
            graph_version="v0",
        )

        history = []
        try:
            stored = await sync_to_async(CHECKPOINTER.load)(graph_context)
            history = load_history(stored)
        except Exception:
            logger.exception(
                "rag_chat.checkpoint_load_failed",
                extra={"thread_id": thread_id},
            )

        loop = asyncio.get_running_loop()

        def stream_callback(text: str) -> None:
            asyncio.run_coroutine_threadsafe(
                self.send_json({"event": "delta", "text": text}), loop
            )

        service = RagQueryService(stream_callback=stream_callback)

        hybrid_config = build_hybrid_config_from_payload(
            payload.model_dump(exclude_none=True)
        )
        try:
            _, result_payload = await sync_to_async(
                service.execute, thread_sensitive=False
            )(
                tool_context=tool_context,
                question=message,
                hybrid=hybrid_config,
                chat_history=list(history),
                graph_state={"question": message, "hybrid": hybrid_config},
            )
        except Exception as exc:
            logger.exception("rag_chat.graph_failed", extra={"thread_id": thread_id})
            await self.send_json({"event": "error", "error": f"Graph error: {exc}"})
            return

        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])
        retrieval_meta = result_payload.get("retrieval") or {}
        passages = build_passage_items_for_workbench(snippets)
        if passages:
            retrieval_meta = dict(retrieval_meta)
            retrieval_meta["passages"] = passages
            result_payload["retrieval"] = retrieval_meta
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
        used_sources = result_payload.get("used_sources")
        if not isinstance(used_sources, list):
            used_sources = []
        if snippet_limit and snippet_limit > 0:
            used_sources = used_sources[:snippet_limit]
        suggested_followups = result_payload.get("suggested_followups")
        if not isinstance(suggested_followups, list):
            suggested_followups = []
        debug_meta = result_payload.get("debug_meta")
        user = self.scope.get("user")
        show_debug = bool(settings.DEBUG) or bool(getattr(user, "is_staff", False))
        if not show_debug:
            debug_meta = None
        else:
            if isinstance(passages, list) and passages:
                debug_meta = dict(debug_meta or {})
                debug_meta["passages"] = passages
            if isinstance(retrieval_meta, dict):
                reference_expansion = retrieval_meta.get("reference_expansion")
                if isinstance(reference_expansion, dict):
                    debug_meta = dict(debug_meta or {})
                    debug_meta["reference_expansion"] = reference_expansion

        await self.send_json(
            {
                "event": "final",
                "answer": answer,
                "snippets": snippet_items,
                "reasoning": reasoning,
                "used_sources": used_sources,
                "suggested_followups": suggested_followups,
                "debug_meta": debug_meta,
                "show_debug": show_debug,
                "thread_id": thread_id,
                "case_id": case_id,
                "collection_id": collection_id,
            }
        )
