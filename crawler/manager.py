from __future__ import annotations

from typing import Any, Mapping

from ai_core.schemas import CrawlerRunRequest
from common.logging import get_logger
from crawler.tasks import crawl_url_task

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
        tenant_id = meta.get("tenant_id")
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

        # Determine overrides that apply to all origins
        ingestion_overrides = {}
        if request.collection_id:
            ingestion_overrides["collection_id"] = request.collection_id
        if request.embedding_profile:
            ingestion_overrides["embedding_profile"] = request.embedding_profile
        if request.scope:
            ingestion_overrides["scope"] = request.scope

        # Iterate through origins (URLs)
        if request.origins:
            for origin in request.origins:
                if not origin.url:
                    continue

                # Dispatch task
                task_result = crawl_url_task.delay(
                    url=origin.url,
                    meta=dict(meta),  # Ensure serializable dict
                    ingestion_overrides=ingestion_overrides,
                )

                dispatched.append({"url": origin.url, "task_id": task_result.id})

        logger.info(
            "crawler_manager.dispatch_completed",
            extra={
                "tenant_id": tenant_id,
                "dispatched_count": len(dispatched),
            },
        )

        return {"status": "dispatched", "count": len(dispatched), "tasks": dispatched}
