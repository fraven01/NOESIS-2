from __future__ import annotations

from django.urls import path

from . import views

app_name = "ai_core"

urlpatterns = [
    path("rag/query/", views.rag_query, name="rag_query"),
    path("rag/documents/upload/", views.rag_upload, name="rag_upload"),
    path("rag/ingestion/run/", views.rag_ingestion_run, name="rag_ingestion_run"),
    path(
        "rag/ingestion/status/",
        views.rag_ingestion_status,
        name="rag_ingestion_status",
    ),
    path("rag/crawler/run/", views.crawler_runner, name="rag_crawler_run"),
    path("rag/feedback/", views.RagFeedbackView.as_view(), name="rag_feedback"),
    path(
        "rag/admin/hard-delete/",
        views.rag_hard_delete_admin,
        name="rag_hard_delete_admin",
    ),
    path("crawl-selected/", views.crawl_selected, name="crawl_selected"),
]
