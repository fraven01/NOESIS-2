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
    path("v1/rag-demo/", views.rag_demo, name="rag_demo"),
]
