from __future__ import annotations

from django.urls import path

from theme import consumers


websocket_urlpatterns = [
    path("ws/rag-tools/chat/", consumers.RagChatConsumer.as_asgi()),
]
