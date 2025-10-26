from django.conf import settings
from django.shortcuts import render
from django.urls import reverse

from ai_core.rag.routing_rules import (
    get_routing_table,
    is_collection_routing_enabled,
)
from structlog.stdlib import get_logger


logger = get_logger(__name__)


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def _tenant_context_from_request(request) -> tuple[str, str]:
    """Return the tenant identifier and schema for the current request."""

    tenant = getattr(request, "tenant", None)
    default_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"

    tenant_id: str | None = None
    tenant_schema: str | None = None

    if tenant is not None:
        tenant_id = getattr(tenant, "tenant_id", None)
        tenant_schema = getattr(tenant, "schema_name", None)

    if not tenant_schema:
        tenant_schema = default_schema

    if not tenant_id:
        tenant_id = tenant_schema

    return tenant_id, tenant_schema


def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""

    tenant_id, tenant_schema = _tenant_context_from_request(request)

    collection_options: list[str] = []
    resolver_profile_hint: str | None = None
    resolver_collection_hint: str | None = None

    try:
        routing_table = get_routing_table()
    except Exception:
        logger.warning("rag_tools.routing_table.unavailable", exc_info=True)
        routing_table = None
    else:
        unique_collections = {
            rule.collection_id
            for rule in routing_table.rules
            if getattr(rule, "collection_id", None)
        }
        collection_options = sorted(
            value for value in unique_collections if isinstance(value, str)
        )

        try:
            resolution = routing_table.resolve_with_metadata(
                tenant=tenant_id,
                process=None,
                collection_id=None,
                workflow_id=None,
                doc_class=None,
            )
        except Exception:
            logger.info("rag_tools.profile_resolution.failed", exc_info=True)
        else:
            resolver_profile_hint = resolution.profile
            if resolution.rule and resolution.rule.collection_id:
                resolver_collection_hint = resolution.rule.collection_id

    return render(
        request,
        "theme/rag_tools.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "default_embedding_profile": getattr(
                settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
            ),
            "collection_options": collection_options,
            "collection_alias_enabled": is_collection_routing_enabled(),
            "resolver_profile_hint": resolver_profile_hint,
            "resolver_collection_hint": resolver_collection_hint,
            "crawler_runner_url": reverse("ai_core:rag_crawler_run"),
            "crawler_default_workflow_id": getattr(
                settings, "CRAWLER_DEFAULT_WORKFLOW_ID", ""
            ),
            "crawler_shadow_default": bool(
                getattr(settings, "CRAWLER_SHADOW_MODE_DEFAULT", False)
            ),
            "crawler_dry_run_default": bool(
                getattr(settings, "CRAWLER_DRY_RUN_DEFAULT", False)
            ),
        },
    )
