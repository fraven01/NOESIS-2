"""URL patterns for documents app."""

from django.urls import path
from . import views

app_name = "documents"

urlpatterns = [
    path("download/<uuid:document_id>/", views.document_download, name="download"),
    path(
        "assets/<uuid:document_id>/<uuid:asset_id>/",
        views.asset_serve,
        name="asset_serve",
    ),
]
