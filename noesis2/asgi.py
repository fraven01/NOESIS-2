"""
ASGI config for noesis2 project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from common.asgi import TenantAuthMiddlewareStack, TenantWebsocketMiddleware

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")

http_application = get_asgi_application()

# Websocket routing loads consumers that touch models, so import after Django setup.
from theme.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": http_application,
        "websocket": TenantWebsocketMiddleware(
            TenantAuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        ),
    }
)
