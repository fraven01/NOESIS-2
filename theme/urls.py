from django.urls import path
from ai_core import views as ai_core_views

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
    path(
        "rag-tools/crawl-selected/", ai_core_views.crawl_selected, name="crawl-selected"
    ),
]
