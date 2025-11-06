from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("rag-tools/", views.rag_tools, name="rag-tools"),
    path("rag-tools/web-search/", views.web_search, name="web-search"),
    path(
        "rag-tools/web-search/ingest-selected/",
        views.web_search_ingest_selected,
        name="web-search-ingest-selected",
    ),
]
