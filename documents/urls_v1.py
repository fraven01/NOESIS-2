"""V1 API routes for documents."""

from __future__ import annotations

from rest_framework.routers import DefaultRouter

from documents.collections_api import DocumentCollectionViewSet

app_name = "documents_v1"

router = DefaultRouter()
router.register(r"collections", DocumentCollectionViewSet, basename="collection")

urlpatterns = router.urls
