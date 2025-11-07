"""URL routes for llm_worker task status polling."""

from django.urls import path

from . import views

urlpatterns = [
    path("tasks/<str:task_id>/", views.task_status, name="task_status"),
]
