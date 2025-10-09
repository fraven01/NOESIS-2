from __future__ import annotations

from django.urls import path

from . import views

app_name = "ai_core"

urlpatterns = [
    path("ping/", views.ping, name="ping"),
    path("intake/", views.intake, name="intake"),
    path("scope/", views.scope, name="scope"),
    path("needs/", views.needs, name="needs"),
    path("sysdesc/", views.sysdesc, name="sysdesc"),
    path("rag/query/", views.rag_query, name="rag_query"),
    path("rag/documents/upload/", views.rag_upload, name="rag_upload"),
    path("rag/ingestion/run/", views.rag_ingestion_run, name="rag_ingestion_run"),
    path(
        "rag/admin/hard-delete/",
        views.rag_hard_delete_admin,
        name="rag_hard_delete_admin",
    ),
    path("v1/rag-demo/", views.rag_demo, name="rag_demo"),
]
