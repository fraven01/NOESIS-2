from __future__ import annotations

from django.urls import path

from . import views

app_name = "ai_core_v1"

urlpatterns = [
    path("ping/", views.ping_v1, name="ping"),
    path("intake/", views.intake_v1, name="intake"),
    path("scope/", views.scope_v1, name="scope"),
    path("needs/", views.needs_v1, name="needs"),
    path("sysdesc/", views.sysdesc_v1, name="sysdesc"),
    path("rag/query/", views.rag_query_v1, name="rag_query"),
    path("rag-demo/", views.rag_demo_v1, name="rag_demo"),
]
