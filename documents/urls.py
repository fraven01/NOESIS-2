"""URL patterns for documents app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views
from .api_views import (
    DocumentCommentViewSet,
    DocumentFavoriteViewSet,
    DocumentNotificationViewSet,
    SavedSearchViewSet,
)

router = DefaultRouter()
router.register(r"favorites", DocumentFavoriteViewSet, basename="document-favorite")
router.register(r"comments", DocumentCommentViewSet, basename="document-comment")
router.register(r"saved-searches", SavedSearchViewSet, basename="document-saved-search")
router.register(
    r"notifications", DocumentNotificationViewSet, basename="document-notification"
)

app_name = "documents"

urlpatterns = [
    path("download/<uuid:document_id>/", views.document_download, name="download"),
    path("recent/", views.recent_documents, name="recent"),
    path("share/<uuid:document_id>/", views.share_document, name="share"),
    path(
        "assets/<uuid:document_id>/<uuid:asset_id>/",
        views.asset_serve,
        name="asset_serve",
    ),
    path("api/", include(router.urls)),
]
