"""In-app notification helpers for document events."""

from __future__ import annotations

from typing import Any

from profiles.models import UserProfile

from .models import DocumentNotification


def _should_notify(user, event_type: str) -> bool:
    if not user:
        return False
    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return False

    if event_type == DocumentNotification.EventType.MENTION:
        return bool(profile.notify_on_mention)

    return True


def create_notification(
    *,
    user,
    event_type: str,
    document=None,
    comment=None,
    payload: dict[str, Any] | None = None,
) -> DocumentNotification | None:
    if not _should_notify(user, event_type):
        return None

    return DocumentNotification.objects.create(
        user=user,
        document=document,
        comment=comment,
        event_type=event_type,
        payload=payload or {},
    )
