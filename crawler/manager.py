from __future__ import annotations

from typing import Any, Mapping

from ai_core.schemas import CrawlerRunRequest, IngestionOverrides
from ai_core.tool_contracts.base import tool_context_from_meta
from common.logging import get_logger
from crawler.tasks import crawl_url_task
from common.celery import with_scope_apply_async

logger = get_logger(__name__)


class CrawlerManager:
    """
    Technical Manager for Crawler operations (Layer 3).
    Orchestrates the dispatch of crawl tasks to workers.
    """

    def dispatch_crawl_request(
        self, request: CrawlerRunRequest, meta: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Dispatch a crawl request to the crawler workers.

        Args:
            request: The validated CrawlerRunRequest schema.
            meta: Context metadata (tenant_id, case_id, etc.).

        Returns:
            Summary of dispatched tasks.
        """
        context = tool_context_from_meta(meta)
        tenant_id = context.scope.tenant_id
        if not tenant_id:
            raise ValueError("Tenant ID is required for crawl dispatch.")

        logger.info(
            "crawler_manager.dispatch_start",
            extra={
                "tenant_id": tenant_id,
                "workflow_id": request.workflow_id,
                "origin_count": len(request.origins or []),
            },
        )

        dispatched = []

        # Validate and build ingestion overrides (BREAKING CHANGE Phase 6)
        overrides_model = IngestionOverrides(
            collection_id=request.collection_id,
            embedding_profile=request.embedding_profile,
            scope=request.scope,
        )
        # Convert to dict for Celery serialization
        ingestion_overrides = overrides_model.model_dump(exclude_none=True)

        # Iterate through origins (URLs)
        if request.origins:
            for origin in request.origins:
                if not origin.url:
                    continue

                url = origin.url
                # Use with_scope_apply_async for better traceability
                # We must break the group into individual calls if we want full trace propagation per task via this helper
                # or we iterate. The helper supports single signature.
                # But the CrawlerManager currently gathers them.
                # If we want to use a group, we might need a `with_scope_group_async` or just iterate.
                # For now, let's iterate to ensure correct context.
                sig = crawl_url_task.s(url=url, meta=dict(meta), ingestion_overrides=ingestion_overrides)
                # context.scope is already in meta, but with_scope_apply_async enforces extraction
                # so we pass scope_dict explicitly to double check.
                scope_dict = context.scope.model_dump(mode="json", exclude_none=True)
                res = with_scope_apply_async(sig, scope_dict)
                
                dispatched.append({
                    "url": url,
                    "task_id": res.id
                })
            
        logger.info(
            "crawler_manager.dispatch_completed",
            extra={
                "tenant_id": tenant_id,
                "dispatched_count": len(dispatched),
            },
        )

        return {"status": "dispatched", "count": len(dispatched), "tasks": dispatched}
