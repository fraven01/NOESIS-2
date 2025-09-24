"""
URL configuration for noesis2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path

from drf_spectacular.views import SpectacularAPIView

from common.views import DemoView
from noesis2.api.docs import (
    BrandedSpectacularRedocView,
    BrandedSpectacularSwaggerView,
)
from users.views import accept_invitation

urlpatterns = [
    path("", include("theme.urls")),
    path("admin/", admin.site.urls),
    path("ai/", include("ai_core.urls")),
    path("v1/ai/", include(("ai_core.urls_v1", "ai_core_v1"), namespace="ai_core_v1")),
    path("api/schema/", SpectacularAPIView.as_view(), name="api-schema"),
    path(
        "api/docs/swagger/",
        BrandedSpectacularSwaggerView.as_view(url_name="api-schema"),
        name="api-docs-swagger",
    ),
    path(
        "api/docs/redoc/",
        BrandedSpectacularRedocView.as_view(url_name="api-schema"),
        name="api-docs-redoc",
    ),
    path("tenant-demo/", DemoView.as_view(), name="tenant-demo"),
    path("invite/accept/<str:token>/", accept_invitation, name="accept-invitation"),
]
