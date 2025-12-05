import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()

from ai_core.services.crawler_runner import run_crawler_runner  # noqa: E402
from ai_core.schemas import CrawlerRunRequest  # noqa: E402
from ai_core.rag.collections import manual_collection_uuid  # noqa: E402
from uuid import uuid4  # noqa: E402

meta = {"tenant_id": "dev", "case_id": "dev-case-test", "trace_id": uuid4().hex}
req = CrawlerRunRequest(
    workflow_id="web-search-ingestion",
    mode="live",
    origins=[{"url": "https://en.wikipedia.org/wiki/Bamberg"}],
    collection_id=str(manual_collection_uuid("dev")),
)
try:
    res = run_crawler_runner(meta=meta, request_model=req, lifecycle_store=None)
    print(f"Status: {res.status_code}")
    origins = res.payload.get("origins", [])
    for o in origins:
        print(f"Origin: {o.get('origin')}")
        print(f"Doc ID: {o.get('document_id')}")
        print(f"Chunks: {o.get('chunk_count')}")
except Exception as e:
    print(f"Error: {e}")
