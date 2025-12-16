from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    # RAG Workbench (Command Center)
    path("rag-tools/", views.workbench_index, name="rag-tools"),
    # Workbench Partials (HTMX)
    path("rag-tools/tool/search/", views.tool_search, name="tool-search"),
    path("rag-tools/tool/ingestion/", views.tool_ingestion, name="tool-ingestion"),
    path("rag-tools/tool/crawler/", views.tool_crawler, name="tool-crawler"),
    path("rag-tools/tool/framework/", views.tool_framework, name="tool-framework"),
    path("rag-tools/tool/chat/", views.tool_chat, name="tool-chat"),
    path(
        "rag-tools/tool/documents/", views.document_explorer, name="document_explorer"
    ),
    # Functional Endpoints
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
    path(
        "rag-tools/chat-submit/",
        views.chat_submit,
        name="chat-submit",
    ),
    path(
        "rag-tools/document-delete/",
        views.document_delete,
        name="document_delete",
    ),
    path(
        "rag-tools/document-restore/",
        views.document_restore,
        name="document_restore",
    ),
]
