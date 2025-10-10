from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("rag-tools/", views.rag_tools, name="rag-tools"),
]
