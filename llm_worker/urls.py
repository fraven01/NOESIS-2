"""URL routes for llm_worker task status polling."""

from django.urls import path

from . import views

app_name = "llm_worker"

urlpatterns = [
    path("run/", views.run_task, name="run_task"),
    path("tasks/<str:task_id>/", views.task_status, name="task_status"),
]
