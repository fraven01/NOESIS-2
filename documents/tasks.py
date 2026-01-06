"""Celery tasks for document upload processing and saved searches."""

from __future__ import annotations

from typing import Any, Dict

from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail
from django.db.models import Q
from django.utils import timezone
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Tenant
from documents.authz import DocumentAuthzService
from documents.notification_dispatcher import (
    external_email_allowed,
    emit_notification_event,
)
from documents.notification_service import create_notification
from documents.upload_worker import UploadWorker
from ai_core.tool_contracts.base import tool_context_from_meta
from common.celery import ScopedTask


@shared_task(base=ScopedTask, queue="ingestion")
def upload_document_task(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    metadata: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Process uploaded document asynchronously.

    Args:
        file_bytes: Raw file content
        filename: Name of the file
        content_type: MIME type
        metadata: Document metadata
        meta: Request context metadata
    """
    # Mock UploadedFile for Worker compatibility
    from io import BytesIO

    class MockUploadedFile:
        def __init__(self, data: bytes, name: str, content_type: str):
            self.file = BytesIO(data)
            self.name = name
            self.content_type = content_type
            self.size = len(data)

        def read(self) -> bytes:
            self.file.seek(0)
            return self.file.read()

    upload = MockUploadedFile(file_bytes, filename, content_type)

    worker = UploadWorker()
    context = tool_context_from_meta(meta)
    user_id = context.scope.user_id
    workflow_id = context.business.workflow_id

    result = worker.process(
        upload,
        tenant_id=context.scope.tenant_id,
        case_id=context.business.case_id,  # BREAKING CHANGE: from business_context
        workflow_id=workflow_id,
        trace_id=context.scope.trace_id,
        invocation_id=context.scope.invocation_id,
        user_id=user_id,
        document_metadata=metadata,
        ingestion_overrides=metadata.get("ingestion_overrides"),
    )

    return {
        "status": result.status,
        "document_id": result.document_id,
        "task_id": result.task_id,
    }


MAX_SEARCHES_PER_RUN = 25
MAX_MATCHES_PER_SEARCH = 50
MAX_EMAIL_DELIVERIES_PER_RUN = 100
MAX_EMAIL_DELIVERY_ATTEMPTS = 5
EMAIL_BACKOFF_BASE_SECONDS = 60


def _apply_saved_search_filters(queryset, saved_search):
    filters = saved_search.filters or {}
    if isinstance(filters, dict):
        case_id = filters.get("case_id")
        if case_id:
            queryset = queryset.filter(case_id=str(case_id))
        source = filters.get("source")
        if source:
            queryset = queryset.filter(source=str(source))
        workflow_id = filters.get("workflow_id")
        if workflow_id:
            queryset = queryset.filter(workflow_id=str(workflow_id))
        collection_id = filters.get("collection_id")
        if collection_id:
            queryset = queryset.filter(
                collection_memberships__collection__collection_id=collection_id
            )
    return queryset


@shared_task(base=ScopedTask)
def run_saved_search_alerts() -> dict[str, int]:
    """Run saved search alerts on a conservative schedule."""

    from documents.models import DocumentNotification, NotificationEvent, SavedSearch

    total_processed = 0
    total_notifications = 0
    now = timezone.now()

    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        tenants = list(Tenant.objects.exclude(schema_name=public_schema))

    for tenant in tenants:
        with schema_context(tenant.schema_name):
            due_searches = (
                SavedSearch.objects.filter(
                    enable_alerts=True,
                    next_run_at__lte=now,
                )
                .select_related("user")
                .order_by("next_run_at")[:MAX_SEARCHES_PER_RUN]
            )

            for saved_search in due_searches:
                queryset = DocumentAuthzService.accessible_documents_queryset(
                    user=saved_search.user,
                    tenant=tenant,
                    permission_type="VIEW",
                )
                if saved_search.last_run_at:
                    queryset = queryset.filter(updated_at__gt=saved_search.last_run_at)
                queryset = _apply_saved_search_filters(queryset, saved_search)
                if saved_search.query:
                    queryset = queryset.filter(
                        Q(
                            metadata__normalized_document__meta__title__icontains=saved_search.query
                        )
                        | Q(external_id__icontains=saved_search.query)
                    )

                matches = list(queryset.order_by("updated_at")[:MAX_MATCHES_PER_SEARCH])

                if matches:
                    create_notification(
                        user=saved_search.user,
                        event_type=DocumentNotification.EventType.SAVED_SEARCH,
                        document=None,
                        comment=None,
                        payload={
                            "saved_search_id": str(saved_search.id),
                            "match_count": len(matches),
                            "document_ids": [str(doc.id) for doc in matches],
                        },
                    )
                    emit_notification_event(
                        user=saved_search.user,
                        event_type=NotificationEvent.EventType.SAVED_SEARCH,
                        document=None,
                        comment=None,
                        payload={
                            "saved_search_id": str(saved_search.id),
                            "match_count": len(matches),
                            "document_ids": [str(doc.id) for doc in matches],
                        },
                    )
                    total_notifications += 1

                saved_search.last_run_at = now
                saved_search.next_run_at = now + timezone.timedelta(
                    hours=(
                        1
                        if saved_search.alert_frequency
                        == SavedSearch.AlertFrequency.HOURLY
                        else (
                            24
                            if saved_search.alert_frequency
                            == SavedSearch.AlertFrequency.DAILY
                            else 24 * 7
                        )
                    )
                )
                saved_search.save(update_fields=["last_run_at", "next_run_at"])
                total_processed += 1

    return {
        "processed": total_processed,
        "notifications": total_notifications,
    }


def _delivery_backoff(attempt: int) -> timezone.timedelta:
    base = EMAIL_BACKOFF_BASE_SECONDS
    delay = base * (2 ** max(0, attempt - 1))
    return timezone.timedelta(seconds=delay)


def _document_title(document) -> str:
    if not document:
        return "document"
    meta = document.metadata or {}
    normalized = meta.get("normalized_document") or {}
    normalized_meta = normalized.get("meta") or {}
    title = normalized_meta.get("title")
    if title:
        return str(title)
    return str(document.id)


def _render_email(event) -> tuple[str, str]:
    document = event.document
    comment = event.comment
    event_type = event.event_type
    title = _document_title(document)

    if event_type == "MENTION":
        subject = f"You were mentioned in {title}"
        body = "You were mentioned in a document comment."
    elif event_type == "SAVED_SEARCH":
        subject = "Saved search updates"
        body = "Your saved search has new matches."
    elif event_type == "COMMENT_REPLY":
        subject = f"New reply in {title}"
        body = "A new reply was posted on a document you follow."
    else:
        subject = "Document notification"
        body = "There is a new document update."

    if comment and comment.text:
        snippet = comment.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = f"{snippet[:197]}..."
        body = f"{body}\n\nComment:\n{snippet}"

    if document:
        body = f"{body}\n\nDocument ID: {document.id}"

    return subject, body


def _render_digest_email(events) -> tuple[str, str]:
    subject = "Daily document notifications"
    lines = ["Here is your daily summary:"]
    for event in events:
        title = _document_title(event.document)
        label = str(event.event_type).replace("_", " ").title()
        lines.append(f"- {label}: {title}")
    return subject, "\n".join(lines)


@shared_task(base=ScopedTask)
def send_pending_email_deliveries() -> dict[str, int]:
    """Send queued external email deliveries with retry backoff."""

    from documents.models import NotificationDelivery
    from profiles.models import UserProfile

    total_sent = 0
    total_failed = 0
    total_skipped = 0
    now = timezone.now()

    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        tenants = list(Tenant.objects.exclude(schema_name=public_schema))

    for tenant in tenants:
        with schema_context(tenant.schema_name):
            deliveries = (
                NotificationDelivery.objects.select_related(
                    "event",
                    "event__user",
                    "event__document",
                    "event__comment",
                )
                .filter(
                    channel=NotificationDelivery.Channel.EMAIL,
                    status=NotificationDelivery.Status.PENDING,
                    next_attempt_at__lte=now,
                )
                .order_by("next_attempt_at")[:MAX_EMAIL_DELIVERIES_PER_RUN]
            )

            daily_groups: dict[int, list[NotificationDelivery]] = {}

            for delivery in deliveries:
                event = delivery.event
                user = event.user
                document = event.document

                if not user or not getattr(user, "email", ""):
                    delivery.status = NotificationDelivery.Status.SKIPPED
                    delivery.last_error = "missing_email"
                    delivery.save(update_fields=["status", "last_error", "updated_at"])
                    total_skipped += 1
                    continue

                try:
                    profile = user.userprofile
                except Exception:
                    delivery.status = NotificationDelivery.Status.SKIPPED
                    delivery.last_error = "profile_missing"
                    delivery.save(update_fields=["status", "last_error", "updated_at"])
                    total_skipped += 1
                    continue

                if not external_email_allowed(profile, event.event_type):
                    delivery.status = NotificationDelivery.Status.SKIPPED
                    delivery.last_error = "external_email_disabled"
                    delivery.save(update_fields=["status", "last_error", "updated_at"])
                    total_skipped += 1
                    continue

                if document:
                    access = DocumentAuthzService.user_can_access_document(
                        user=user,
                        document=document,
                        permission_type="VIEW",
                        tenant=tenant,
                    )
                    if not access.allowed:
                        delivery.status = NotificationDelivery.Status.SKIPPED
                        delivery.last_error = "permission_denied"
                        delivery.save(
                            update_fields=["status", "last_error", "updated_at"]
                        )
                        total_skipped += 1
                        continue

                if (
                    profile.external_email_frequency
                    == UserProfile.ExternalEmailFrequency.DAILY
                ):
                    daily_groups.setdefault(user.id, []).append(delivery)
                    continue

                delivery.attempts += 1
                delivery.save(update_fields=["attempts", "updated_at"])

                subject, body = _render_email(event)
                from_email = getattr(
                    settings,
                    "DEFAULT_FROM_EMAIL",
                    "no-reply@noesis.local",
                )
                try:
                    send_mail(
                        subject=subject,
                        message=body,
                        from_email=from_email,
                        recipient_list=[user.email],
                        fail_silently=False,
                    )
                except Exception as exc:
                    delivery.last_error = str(exc)
                    if delivery.attempts >= MAX_EMAIL_DELIVERY_ATTEMPTS:
                        delivery.status = NotificationDelivery.Status.FAILED
                        total_failed += 1
                    else:
                        delivery.next_attempt_at = now + _delivery_backoff(
                            delivery.attempts
                        )
                    delivery.save(
                        update_fields=[
                            "status",
                            "last_error",
                            "next_attempt_at",
                            "updated_at",
                        ]
                    )
                    continue

                delivery.status = NotificationDelivery.Status.SENT
                delivery.sent_at = now
                delivery.save(update_fields=["status", "sent_at", "updated_at"])
                total_sent += 1

            for deliveries_for_user in daily_groups.values():
                if not deliveries_for_user:
                    continue
                user = deliveries_for_user[0].event.user
                if not user:
                    for delivery in deliveries_for_user:
                        delivery.status = NotificationDelivery.Status.SKIPPED
                        delivery.last_error = "missing_email"
                        delivery.save(
                            update_fields=["status", "last_error", "updated_at"]
                        )
                        total_skipped += 1
                    continue

                for delivery in deliveries_for_user:
                    delivery.attempts += 1
                    delivery.save(update_fields=["attempts", "updated_at"])

                events = [delivery.event for delivery in deliveries_for_user]
                subject, body = _render_digest_email(events)
                from_email = getattr(
                    settings,
                    "DEFAULT_FROM_EMAIL",
                    "no-reply@noesis.local",
                )
                try:
                    send_mail(
                        subject=subject,
                        message=body,
                        from_email=from_email,
                        recipient_list=[user.email],
                        fail_silently=False,
                    )
                except Exception as exc:
                    for delivery in deliveries_for_user:
                        delivery.last_error = str(exc)
                        if delivery.attempts >= MAX_EMAIL_DELIVERY_ATTEMPTS:
                            delivery.status = NotificationDelivery.Status.FAILED
                            total_failed += 1
                        else:
                            delivery.next_attempt_at = now + _delivery_backoff(
                                delivery.attempts
                            )
                        delivery.save(
                            update_fields=[
                                "status",
                                "last_error",
                                "next_attempt_at",
                                "updated_at",
                            ]
                        )
                    continue

                for delivery in deliveries_for_user:
                    delivery.status = NotificationDelivery.Status.SENT
                    delivery.sent_at = now
                    delivery.save(update_fields=["status", "sent_at", "updated_at"])
                total_sent += len(deliveries_for_user)

    return {
        "sent": total_sent,
        "failed": total_failed,
        "skipped": total_skipped,
    }
