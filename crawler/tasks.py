from typing import Any, Mapping

from celery import shared_task

from common.logging import get_logger
from crawler.fetcher import FetchRequest
from crawler.http_fetcher import HttpFetcher
from crawler.worker import CrawlerWorker
from ai_core.tool_contracts.base import tool_context_from_meta

logger = get_logger(__name__)


@shared_task(bind=True, queue="crawler", name="crawler.tasks.crawl_url_task")
def crawl_url_task(
    self,
    url: str,
    meta: Mapping[str, Any],
    ingestion_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a single URL crawl using the CrawlerWorker.

    Args:
        url: The URL to crawl.
        meta: Context metadata (tenant_id, case_id, etc.).
        ingestion_overrides: Optional overrides for the ingestion process.
    """
    context = tool_context_from_meta(meta)
    tenant_id = context.scope.tenant_id
    if not tenant_id:
        logger.error("crawl_url_task.missing_tenant_id", extra={"url": url})
        return {"status": "error", "reason": "missing_tenant_id"}

    logger.info(
        "crawl_url_task.start",
        extra={"url": url, "tenant_id": tenant_id, "task_id": self.request.id},
    )

    try:
        # 1. Setup Worker
        # We use HttpFetcher by default for web crawling
        fetcher = HttpFetcher(config={})
        worker = CrawlerWorker(
            fetcher=fetcher,
            # using default ingestion_task which is run_ingestion_graph
            # and default blob_writer_factory
        )

        # 2. Prepare Request
        # In a real scenario, politeness/limits would come from meta/config
        from urllib.parse import urlparse
        from crawler.fetcher import PolitenessContext

        parsed = urlparse(url)
        host = parsed.hostname or url

        politeness = PolitenessContext(
            host=host, user_agent="noesis-crawler/1.0-worker"
        )
        request = FetchRequest(
            canonical_source=url, politeness=politeness, metadata=meta
        )

        # 3. Process
        # worker.process handles fetching, asset extraction, and dispatching ingestion
        # BREAKING CHANGE (Option A): Business IDs from business_context
        doc_meta = {"source": url}
        if context.business.workflow_id:
            doc_meta["workflow_id"] = context.business.workflow_id
        if context.business.collection_id:
            doc_meta["collection_id"] = context.business.collection_id

        result = worker.process(
            request,
            tenant_id=tenant_id,
            case_id=context.business.case_id,
            crawl_id=meta.get("crawl_id"),
            trace_id=context.scope.trace_id,
            document_metadata=doc_meta,
            ingestion_overrides=ingestion_overrides,
            meta_overrides=meta,
        )

        logger.info(
            "crawl_url_task.completed",
            extra={
                "url": url,
                "status": result.status,
                "ingestion_task_id": result.task_id,
            },
        )

        return {
            "status": result.status,
            "ingestion_task_id": result.task_id,
            "fetch_status": result.fetch_result.status.value,
        }

    except Exception as exc:
        logger.exception("crawl_url_task.failed", extra={"url": url})
        return {"status": "error", "reason": str(exc)}
