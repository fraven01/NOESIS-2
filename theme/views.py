from django.conf import settings
from django.shortcuts import render


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""

    host = request.get_host() or ""
    hostname = host.split(":", maxsplit=1)[0]

    tenant_id = hostname or "dev.localhost"

    tenant_schema = None
    if hostname and "." in hostname:
        candidate_schema = hostname.split(".", maxsplit=1)[0] or None
        if candidate_schema and any(char.isalpha() for char in candidate_schema):
            tenant_schema = candidate_schema

    if not tenant_schema:
        tenant_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"

    return render(
        request,
        "theme/rag_tools.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "default_embedding_profile": getattr(
                settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
            ),
        },
    )
