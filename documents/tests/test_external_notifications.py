from __future__ import annotations

from unittest.mock import Mock

import pytest
from django.core import mail
from django.utils import timezone
from rest_framework.test import APIClient

from customers.models import Tenant
from documents.models import (
    Document,
    DocumentPermission,
    NotificationDelivery,
    NotificationEvent,
)
from documents.notification_dispatcher import emit_notification_event
from documents.tasks import send_pending_email_deliveries
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


def test_comment_mention_emits_notification_event(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    author = UserFactory()
    mentioned = UserFactory(username="mention-target")

    document = Document.objects.create(
        tenant=tenant,
        hash="mention-hash",
        source="upload",
        metadata={},
        created_by=author,
    )

    DocumentPermission.objects.create(
        document=document,
        user=mentioned,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    client = APIClient()
    client.force_authenticate(user=author)

    response = client.post(
        "/documents/api/comments/",
        {"document": str(document.id), "text": f"Hi <@{mentioned.id}>"},
        format="json",
        HTTP_X_TENANT_ID=tenant.schema_name,
    )

    assert response.status_code == 201
    comment_id = int(response.data["id"])

    event = NotificationEvent.objects.filter(
        user=mentioned,
        event_type=NotificationEvent.EventType.MENTION,
    ).first()
    assert event is not None
    assert event.document_id == document.id
    assert event.comment_id == comment_id
    assert event.payload.get("actor_user_id") == str(author.id)


def test_dispatcher_creates_delivery_when_allowed(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    recipient = UserFactory()
    profile = recipient.userprofile
    profile.external_email_enabled = True
    profile.notify_on_mention = True
    profile.save()

    document = Document.objects.create(
        tenant=tenant,
        hash="dispatch-hash",
        source="upload",
        metadata={},
    )

    DocumentPermission.objects.create(
        document=document,
        user=recipient,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    event = emit_notification_event(
        user=recipient,
        event_type=NotificationEvent.EventType.MENTION,
        document=document,
        comment=None,
        payload={},
    )

    assert event is not None
    assert NotificationDelivery.objects.filter(event=event).exists()


def test_dispatcher_respects_notify_flags(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    recipient = UserFactory()
    profile = recipient.userprofile
    profile.external_email_enabled = True
    profile.notify_on_mention = False
    profile.notify_on_comment_reply = False
    profile.save()

    document = Document.objects.create(
        tenant=tenant,
        hash="flags-hash",
        source="upload",
        metadata={},
    )

    DocumentPermission.objects.create(
        document=document,
        user=recipient,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )

    mention_event = emit_notification_event(
        user=recipient,
        event_type=NotificationEvent.EventType.MENTION,
        document=document,
        comment=None,
        payload={},
    )
    assert mention_event is not None
    assert not NotificationDelivery.objects.filter(event=mention_event).exists()

    reply_event = emit_notification_event(
        user=recipient,
        event_type=NotificationEvent.EventType.COMMENT_REPLY,
        document=document,
        comment=None,
        payload={},
    )
    assert reply_event is not None
    assert not NotificationDelivery.objects.filter(event=reply_event).exists()


def test_dispatcher_blocks_event_without_permission(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    recipient = UserFactory()
    profile = recipient.userprofile
    profile.external_email_enabled = True
    profile.notify_on_mention = True
    profile.save()

    document = Document.objects.create(
        tenant=tenant,
        hash="noaccess-hash",
        source="upload",
        metadata={},
    )

    event = emit_notification_event(
        user=recipient,
        event_type=NotificationEvent.EventType.MENTION,
        document=document,
        comment=None,
        payload={},
    )

    assert event is None
    assert NotificationDelivery.objects.count() == 0


def _create_pending_delivery(tenant, user):
    document = Document.objects.create(
        tenant=tenant,
        hash="delivery-hash",
        source="upload",
        metadata={},
    )
    DocumentPermission.objects.create(
        document=document,
        user=user,
        permission_type=DocumentPermission.PermissionType.VIEW,
    )
    event = NotificationEvent.objects.create(
        user=user,
        document=document,
        event_type=NotificationEvent.EventType.MENTION,
        payload={},
    )
    delivery = NotificationDelivery.objects.create(
        event=event,
        channel=NotificationDelivery.Channel.EMAIL,
        status=NotificationDelivery.Status.PENDING,
        next_attempt_at=timezone.now() - timezone.timedelta(minutes=1),
    )
    return delivery


def test_send_pending_email_deliveries_marks_sent(
    settings, test_tenant_schema_name
):
    settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    recipient = UserFactory(email="notify@example.com")
    profile = recipient.userprofile
    profile.external_email_enabled = True
    profile.notify_on_mention = True
    profile.save()

    delivery = _create_pending_delivery(tenant, recipient)

    result = send_pending_email_deliveries()
    delivery.refresh_from_db()

    assert result["sent"] >= 1
    assert delivery.status == NotificationDelivery.Status.SENT
    assert delivery.sent_at is not None
    assert delivery.attempts == 1
    assert len(mail.outbox) == 1


def test_send_pending_email_deliveries_failure_sets_backoff(
    monkeypatch, test_tenant_schema_name
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    recipient = UserFactory(email="notify-fail@example.com")
    profile = recipient.userprofile
    profile.external_email_enabled = True
    profile.notify_on_mention = True
    profile.save()

    delivery = _create_pending_delivery(tenant, recipient)

    monkeypatch.setattr(
        "documents.tasks.send_mail",
        Mock(side_effect=RuntimeError("SMTP down")),
    )

    result = send_pending_email_deliveries()
    delivery.refresh_from_db()

    assert result["failed"] == 0
    assert delivery.status == NotificationDelivery.Status.PENDING
    assert delivery.attempts == 1
    assert delivery.next_attempt_at > timezone.now()
    assert delivery.last_error


def test_send_pending_email_deliveries_skips_without_permission(
    test_tenant_schema_name,
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    recipient = UserFactory(email="notify-skip@example.com")
    profile = recipient.userprofile
    profile.external_email_enabled = True
    profile.notify_on_mention = True
    profile.save()

    document = Document.objects.create(
        tenant=tenant,
        hash="skip-hash",
        source="upload",
        metadata={},
    )
    event = NotificationEvent.objects.create(
        user=recipient,
        document=document,
        event_type=NotificationEvent.EventType.MENTION,
        payload={},
    )
    delivery = NotificationDelivery.objects.create(
        event=event,
        channel=NotificationDelivery.Channel.EMAIL,
        status=NotificationDelivery.Status.PENDING,
        next_attempt_at=timezone.now() - timezone.timedelta(minutes=1),
    )

    result = send_pending_email_deliveries()
    delivery.refresh_from_db()

    assert result["skipped"] >= 1
    assert delivery.status == NotificationDelivery.Status.SKIPPED
    assert delivery.last_error == "permission_denied"
