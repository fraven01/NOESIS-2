"""URL patterns for documents app."""
from django.urls import path
from . import views

app_name = 'documents'

urlpatterns = [
    path('download/<uuid:document_id>/', views.document_download, name='download'),
]
