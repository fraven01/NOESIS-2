"""REST API endpoints for document collaboration features (Phase 4a)."""

from __future__ import annotations

from django.db import transaction
from django.utils import timezone
from drf_spectacular.utils import extend_schema_view
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.response import Response

from customers.tenant_context import TenantContext
from documents.authz import DocumentAuthzService
from documents.mentions import resolve_mentioned_users
from documents.notification_dispatcher import emit_notification_event
from documents.notification_service import create_notification
from documents.serializers import (
    DocumentCommentSerializer,
    DocumentCommentResponseSerializer,
    DocumentFavoriteSerializer,
    DocumentFavoriteResponseSerializer,
    DocumentNotificationSerializer,
    DocumentNotificationStatusSerializer,
    SavedSearchResponseSerializer,
    SavedSearchSerializer,
)
from documents.throttles import CommentCreateThrottle
from documents.models import (
    Document,
    DocumentComment,
    DocumentMention,
    DocumentNotification,
    DocumentSubscription,
    NotificationEvent,
    SavedSearch,
    UserDocumentFavorite,
)
from noesis2.api import JSON_ERROR_STATUSES, default_extend_schema
from profiles.models import UserProfile


def _require_user(request):
    user = getattr(request, "user", None)
    if not user or not user.is_authenticated:
        raise PermissionDenied("Authentication required")
    return user


def _tenant_from_request(request):
    tenant = getattr(request, "tenant", None)
    if tenant is None:
        tenant = TenantContext.from_request(request, allow_headers=True, require=True)
    return tenant


def _is_tenant_admin(user) -> bool:
    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return False
    return profile.is_active and profile.role == UserProfile.Roles.TENANT_ADMIN


def _next_run_at(now, frequency: str) -> timezone.datetime:
    if frequency == SavedSearch.AlertFrequency.DAILY:
        return now + timezone.timedelta(days=1)
    if frequency == SavedSearch.AlertFrequency.WEEKLY:
        return now + timezone.timedelta(days=7)
    return now + timezone.timedelta(hours=1)


@extend_schema_view(
    list=default_extend_schema(),
    retrieve=default_extend_schema(),
    create=default_extend_schema(
        responses={201: DocumentFavoriteResponseSerializer},
        error_statuses=JSON_ERROR_STATUSES,
    ),
    update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    partial_update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    destroy=default_extend_schema(),
)
class DocumentFavoriteViewSet(viewsets.ModelViewSet):
    serializer_class = DocumentFavoriteSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return UserDocumentFavorite.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        if isinstance(response.data, dict):
            response.data = {"idempotent": False, **response.data}
        return response

    def perform_create(self, serializer):
        user = _require_user(self.request)
        document = serializer.validated_data["document"]
        tenant = _tenant_from_request(self.request)

        access = DocumentAuthzService.user_can_access_document(
            user=user,
            document=document,
            permission_type="VIEW",
            tenant=tenant,
        )
        if not access.allowed:
            raise PermissionDenied("Permission denied")

        serializer.save(user=user)


@extend_schema_view(
    list=default_extend_schema(),
    retrieve=default_extend_schema(),
    create=default_extend_schema(
        responses={201: DocumentCommentResponseSerializer},
        error_statuses=JSON_ERROR_STATUSES,
    ),
    update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    partial_update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    destroy=default_extend_schema(),
)
class DocumentCommentViewSet(viewsets.ModelViewSet):
    serializer_class = DocumentCommentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_throttles(self):
        if self.action == "create":
            return [CommentCreateThrottle()]
        return []

    def get_queryset(self):
        user = _require_user(self.request)
        doc_id = self.request.query_params.get("document_id")
        base_qs = DocumentComment.objects.select_related("user", "document", "parent")
        if self.action in ("list",):
            if not doc_id:
                raise ValidationError(
                    {"document_id": "This query parameter is required."}
                )
            document = Document.objects.filter(id=doc_id).first()
            if document is None:
                return base_qs.none()
            tenant = _tenant_from_request(self.request)
            access = DocumentAuthzService.user_can_access_document(
                user=user,
                document=document,
                permission_type="VIEW",
                tenant=tenant,
            )
            if not access.allowed:
                raise PermissionDenied("Permission denied")
            return base_qs.filter(document=document)

        return base_qs

    def get_object(self):
        obj = super().get_object()
        user = _require_user(self.request)
        tenant = _tenant_from_request(self.request)
        access = DocumentAuthzService.user_can_access_document(
            user=user,
            document=obj.document,
            permission_type="VIEW",
            tenant=tenant,
        )
        if not access.allowed:
            raise PermissionDenied("Permission denied")
        return obj

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        if isinstance(response.data, dict):
            response.data = {"idempotent": False, **response.data}
        return response

    def perform_create(self, serializer):
        user = _require_user(self.request)
        document = serializer.validated_data["document"]
        tenant = _tenant_from_request(self.request)

        access = DocumentAuthzService.user_can_access_document(
            user=user,
            document=document,
            permission_type="COMMENT",
            tenant=tenant,
        )
        if not access.allowed:
            raise PermissionDenied("Permission denied")

        with transaction.atomic():
            comment = serializer.save(user=user)
            DocumentSubscription.objects.get_or_create(
                user=user,
                document=document,
            )

            mentioned_users = resolve_mentioned_users(comment.text)
            mentions = []
            for mentioned_user in mentioned_users:
                if mentioned_user.id == user.id:
                    continue
                mention_access = DocumentAuthzService.user_can_access_document(
                    user=mentioned_user,
                    document=document,
                    permission_type="VIEW",
                    tenant=tenant,
                )
                if not mention_access.allowed:
                    continue
                mentions.append(
                    DocumentMention(comment=comment, mentioned_user=mentioned_user)
                )
                create_notification(
                    user=mentioned_user,
                    event_type=DocumentNotification.EventType.MENTION,
                    document=document,
                    comment=comment,
                    payload={
                        "comment_id": str(comment.id),
                        "actor_user_id": str(user.id),
                    },
                )
                emit_notification_event(
                    user=mentioned_user,
                    event_type=NotificationEvent.EventType.MENTION,
                    document=document,
                    comment=comment,
                    payload={
                        "comment_id": str(comment.id),
                        "actor_user_id": str(user.id),
                    },
                )

            if mentions:
                DocumentMention.objects.bulk_create(mentions, ignore_conflicts=True)

            if comment.parent_id:
                subscribers = (
                    DocumentSubscription.objects.select_related("user")
                    .filter(document=document)
                    .exclude(user_id=user.id)
                )
                for subscription in subscribers:
                    emit_notification_event(
                        user=subscription.user,
                        event_type=NotificationEvent.EventType.COMMENT_REPLY,
                        document=document,
                        comment=comment,
                        payload={
                            "comment_id": str(comment.id),
                            "actor_user_id": str(user.id),
                        },
                    )

    def perform_update(self, serializer):
        user = _require_user(self.request)
        instance = serializer.instance
        if instance.user_id != user.id and not _is_tenant_admin(user):
            raise PermissionDenied("Only the author can edit this comment.")
        serializer.save()

    def perform_destroy(self, instance):
        user = _require_user(self.request)
        if instance.user_id != user.id and not _is_tenant_admin(user):
            raise PermissionDenied("Only the author can delete this comment.")
        instance.delete()


@extend_schema_view(
    list=default_extend_schema(),
    retrieve=default_extend_schema(),
    create=default_extend_schema(
        responses={201: SavedSearchResponseSerializer},
        error_statuses=JSON_ERROR_STATUSES,
    ),
    update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    partial_update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    destroy=default_extend_schema(),
)
class SavedSearchViewSet(viewsets.ModelViewSet):
    serializer_class = SavedSearchSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SavedSearch.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        if isinstance(response.data, dict):
            response.data = {"idempotent": False, **response.data}
        return response

    def perform_create(self, serializer):
        user = _require_user(self.request)
        now = timezone.now()
        frequency = serializer.validated_data.get(
            "alert_frequency", SavedSearch.AlertFrequency.HOURLY
        )
        serializer.save(
            user=user,
            next_run_at=_next_run_at(now, frequency),
        )

    def perform_update(self, serializer):
        now = timezone.now()
        instance = serializer.instance
        frequency = serializer.validated_data.get(
            "alert_frequency", instance.alert_frequency
        )
        enable_alerts = serializer.validated_data.get(
            "enable_alerts", instance.enable_alerts
        )
        next_run_at = instance.next_run_at
        if enable_alerts and (
            frequency != instance.alert_frequency or not instance.enable_alerts
        ):
            next_run_at = _next_run_at(now, frequency)
        serializer.save(next_run_at=next_run_at)


@extend_schema_view(
    list=default_extend_schema(),
    retrieve=default_extend_schema(),
    partial_update=default_extend_schema(error_statuses=JSON_ERROR_STATUSES),
    mark_all_read=default_extend_schema(
        responses={200: DocumentNotificationStatusSerializer},
        error_statuses=JSON_ERROR_STATUSES,
    ),
)
class DocumentNotificationViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = DocumentNotificationSerializer
    permission_classes = [permissions.IsAuthenticated]
    http_method_names = ["get", "patch", "post", "head", "options"]

    def get_queryset(self):
        return DocumentNotification.objects.filter(user=self.request.user)

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.read_at is None:
            instance.read_at = timezone.now()
            instance.save(update_fields=["read_at"])
        serializer = self.get_serializer(instance)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=["post"])
    def mark_all_read(self, request):
        user = _require_user(request)
        now = timezone.now()
        DocumentNotification.objects.filter(user=user, read_at__isnull=True).update(
            read_at=now
        )
        return Response({"status": "ok", "idempotent": False})
