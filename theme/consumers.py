from __future__ import annotations

import asyncio
from uuid import uuid4

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from pydantic import ValidationError
from structlog.stdlib import get_logger

from ai_core.graph.core import GraphContext, ThreadAwareCheckpointer
from ai_core.graphs.technical.retrieval_augmented_generation import (
    RAG_IO_VERSION_STRING,
    RAG_SCHEMA_ID,
    run as run_rag_graph,
)
from ai_core.tool_contracts import ContextError
from theme.chat_utils import (
    append_history,
    build_hybrid_config_from_payload,
    build_snippet_items,
    load_history,
    resolve_history_limit,
    trim_history,
)
from theme.websocket_payloads import RagChatPayload
from theme.websocket_utils import build_websocket_context


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
        case_id = payload.case_id or "dev-case-local"
        collection_id = payload.collection_id
        thread_id = payload.thread_id or uuid4().hex
        if payload.global_search:
            case_id = None

        try:
            scope, business = build_websocket_context(
                request=self.scope,
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

        tool_context = scope.to_tool_context(
            business=business,
            metadata={"graph_name": "rag.default", "graph_version": "v0"},
        )

        meta = {
            "scope_context": scope.model_dump(mode="json", exclude_none=True),
            "business_context": business.model_dump(mode="json", exclude_none=True),
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

        graph_context = GraphContext(
            tool_context=tool_context,
            graph_name="rag.default",
            graph_version="v0",
        )

        history_limit = resolve_history_limit()
        history = []
        try:
            stored = await sync_to_async(CHECKPOINTER.load)(graph_context)
            history = load_history(stored)
        except Exception:
            logger.exception(
                "rag_chat.checkpoint_load_failed",
                extra={"thread_id": thread_id},
            )

        state = {
            "schema_id": RAG_SCHEMA_ID,
            "schema_version": RAG_IO_VERSION_STRING,
            "question": message,
            "query": message,
            "hybrid": build_hybrid_config_from_payload(
                payload.model_dump(exclude_none=True)
            ),
            "chat_history": list(history),
        }

        loop = asyncio.get_running_loop()

        def stream_callback(text: str) -> None:
            asyncio.run_coroutine_threadsafe(
                self.send_json({"event": "delta", "text": text}), loop
            )

        meta["stream_callback"] = stream_callback

        try:
            final_state, result_payload = await sync_to_async(run_rag_graph)(
                state, meta
            )
        except Exception as exc:
            logger.exception("rag_chat.graph_failed", extra={"thread_id": thread_id})
            await self.send_json({"event": "error", "error": f"Graph error: {exc}"})
            return

        answer = result_payload.get("answer", "No answer generated.")
        snippets = result_payload.get("snippets", [])
        snippet_items = build_snippet_items(snippets)

        append_history(history, role="user", content=message)
        append_history(history, role="assistant", content=answer)
        history = trim_history(history, limit=history_limit)
        try:
            await sync_to_async(CHECKPOINTER.save)(
                graph_context, {"chat_history": history}
            )
        except Exception:
            logger.exception(
                "rag_chat.checkpoint_save_failed",
                extra={"thread_id": thread_id},
            )

        await self.send_json(
            {
                "event": "final",
                "answer": answer,
                "snippets": snippet_items,
                "thread_id": thread_id,
                "case_id": case_id,
                "collection_id": collection_id,
            }
        )
