"""Dispatcher for external notification events and deliveries."""

from __future__ import annotations

from django.utils import timezone

from documents.authz import DocumentAuthzService
from documents.models import NotificationDelivery, NotificationEvent
from profiles.models import UserProfile


def external_email_allowed(profile: UserProfile, event_type: str) -> bool:
    if not profile.external_email_enabled:
        return False
    if event_type == NotificationEvent.EventType.MENTION:
        return profile.notify_on_mention
    if event_type == NotificationEvent.EventType.COMMENT_REPLY:
        return profile.notify_on_comment_reply
    return True


def _next_daily_window(now: timezone.datetime) -> timezone.datetime:
    next_day = now + timezone.timedelta(days=1)
    return next_day.replace(hour=0, minute=0, second=0, microsecond=0)


def _next_delivery_at(
    profile: UserProfile, now: timezone.datetime
) -> timezone.datetime:
    if profile.external_email_frequency == UserProfile.ExternalEmailFrequency.DAILY:
        return _next_daily_window(now)
    return now


def create_delivery_for_email(
    *,
    event: NotificationEvent,
    user,
    profile: UserProfile,
) -> NotificationDelivery | None:
    if not external_email_allowed(profile, event.event_type):
        return None
    if not getattr(user, "email", ""):
        return None

    now = timezone.now()
    delivery_at = _next_delivery_at(profile, now)
    delivery, _ = NotificationDelivery.objects.get_or_create(
        event=event,
        channel=NotificationDelivery.Channel.EMAIL,
        defaults={
            "status": NotificationDelivery.Status.PENDING,
            "next_attempt_at": delivery_at,
        },
    )
    if delivery.status == NotificationDelivery.Status.PENDING:
        if delivery.next_attempt_at != delivery_at:
            delivery.next_attempt_at = delivery_at
            delivery.save(update_fields=["next_attempt_at", "updated_at"])
    return delivery


def emit_notification_event(
    *,
    user,
    event_type: str,
    document=None,
    comment=None,
    payload: dict | None = None,
) -> NotificationEvent | None:
    if not user:
        return None
    if document:
        access = DocumentAuthzService.user_can_access_document(
            user=user,
            document=document,
            permission_type="VIEW",
            tenant=getattr(document, "tenant", None),
        )
        if not access.allowed:
            return None

    event = NotificationEvent.objects.create(
        user=user,
        document=document,
        comment=comment,
        event_type=event_type,
        payload=payload or {},
    )

    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return event

    create_delivery_for_email(event=event, user=user, profile=profile)
    return event
