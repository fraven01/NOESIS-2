"""URL configuration for cases app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from cases.api import CaseViewSet

app_name = "cases"

router = DefaultRouter()
router.register(r"", CaseViewSet, basename="case")

urlpatterns = [
    path("", include(router.urls)),
]
