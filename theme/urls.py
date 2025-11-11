from django.urls import path

from . import dev_hitl_views, views

urlpatterns = [
    path("", views.home, name="home"),
    path("rag-tools/", views.rag_tools, name="rag-tools"),
    path("rag-tools/web-search/", views.web_search, name="web-search"),
    path(
        "rag-tools/web-search/ingest-selected/",
        views.web_search_ingest_selected,
        name="web-search-ingest-selected",
    ),
    path(
        "dev/hitl/runs/<slug:run_id>/",
        dev_hitl_views.get_run_payload,
        name="dev-hitl-run-api",
    ),
    path(
        "dev/hitl/approve-candidates/",
        dev_hitl_views.approve_candidates,
        name="dev-hitl-approve",
    ),
    path(
        "dev/hitl/progress/<slug:run_id>/stream/",
        dev_hitl_views.progress_stream,
        name="dev-hitl-progress",
    ),
    path("dev/hitl/", dev_hitl_views.dev_hitl_page, name="dev-hitl"),
    path("dev/hitl/<slug:run_id>/", dev_hitl_views.dev_hitl_page, name="dev-hitl-run"),
]
