from django.shortcuts import render


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def rag_tools(request):
    """Render a minimal interface to exercise the RAG endpoints manually."""

    host = request.get_host() or ""
    tenant_id = host.split(":", maxsplit=1)[0] if host else "dev.localhost"

    return render(
        request,
        "theme/rag_tools.html",
        {
            "tenant_id": tenant_id,
        },
    )
