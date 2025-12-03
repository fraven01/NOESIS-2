from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("rag-tools/", views.rag_tools, name="rag-tools"),
    path("document-space/", views.document_space, name="document-space"),
    path("rag-tools/web-search/", views.web_search, name="web-search"),
    path(
        "rag-tools/web-search/ingest-selected/",
        views.web_search_ingest_selected,
        name="web-search-ingest-selected",
    ),
    path(
        "rag-tools/crawler-submit/",
        views.crawler_submit,
        name="crawler-submit",
    ),
    path(
        "rag-tools/ingestion-submit/",
        views.ingestion_submit,
        name="ingestion-submit",
    ),
    path(
        "rag-tools/start-rerank-workflow/",
        views.start_rerank_workflow,
        name="rag_tools_start_rerank",
    ),
    path(
        "framework-analysis/",
        views.framework_analysis_tool,
        name="framework-analysis-tool",
    ),
    path(
        "framework-analysis/submit/",
        views.framework_analysis_submit,
        name="framework-analysis-submit",
    ),
]
