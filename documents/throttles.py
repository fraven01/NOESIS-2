"""Rate limiting helpers for document APIs."""

from __future__ import annotations

from django.conf import settings
from rest_framework.throttling import UserRateThrottle


class CommentCreateThrottle(UserRateThrottle):
    rate = getattr(settings, "DOCUMENT_COMMENT_CREATE_RATE", "60/min")
