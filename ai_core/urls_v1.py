from __future__ import annotations

from django.urls import path

from . import views
from .views_framework import FrameworkAnalysisView

app_name = "ai_core_v1"

urlpatterns = [
    path("ping/", views.ping_v1, name="ping"),
    path("intake/", views.intake_v1, name="intake"),
    path("rag/query/", views.rag_query_v1, name="rag_query"),
    path(
        "frameworks/analyze/", FrameworkAnalysisView.as_view(), name="framework_analyze"
    ),
]
