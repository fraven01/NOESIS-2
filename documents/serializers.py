"""Serializers for document API responses."""

from __future__ import annotations

from rest_framework import serializers

from documents.models import (
    Document,
    DocumentComment,
    DocumentCollection,
    DocumentMention,
    DocumentNotification,
    SavedSearch,
    UserDocumentFavorite,
)


class DocumentSerializer(serializers.ModelSerializer):
    created_by = serializers.SerializerMethodField()
    updated_by = serializers.SerializerMethodField()
    document_id = serializers.UUIDField(source="id", read_only=True)
    tenant_id = serializers.CharField(read_only=True)

    class Meta:
        model = Document
        fields = [
            "document_id",
            "tenant_id",
            "source",
            "hash",
            "metadata",
            "lifecycle_state",
            "lifecycle_updated_at",
            "created_at",
            "updated_at",
            "created_by",
            "updated_by",
        ]

    def get_created_by(self, obj):
        user = getattr(obj, "created_by", None)
        if not user:
            return None
        full_name = f"{user.first_name} {user.last_name}".strip()
        return {
            "id": user.pk,
            "username": user.username,
            "full_name": full_name or None,
        }

    def get_updated_by(self, obj):
        user = getattr(obj, "updated_by", None)
        if not user:
            return None
        return {
            "id": user.pk,
            "username": user.username,
        }


class DocumentFavoriteSerializer(serializers.ModelSerializer):
    document = serializers.PrimaryKeyRelatedField(
        queryset=Document.objects.all(), write_only=True
    )
    document_id = serializers.UUIDField(source="document.id", read_only=True)

    class Meta:
        model = UserDocumentFavorite
        fields = ["id", "document", "document_id", "favorited_at"]
        read_only_fields = ["id", "favorited_at", "document_id"]


class DocumentFavoriteResponseSerializer(DocumentFavoriteSerializer):
    idempotent = serializers.BooleanField(read_only=True)

    class Meta(DocumentFavoriteSerializer.Meta):
        fields = ["idempotent", *DocumentFavoriteSerializer.Meta.fields]
        read_only_fields = [
            *DocumentFavoriteSerializer.Meta.read_only_fields,
            "idempotent",
        ]


class DocumentCommentSerializer(serializers.ModelSerializer):
    document = serializers.PrimaryKeyRelatedField(
        queryset=Document.objects.all(), write_only=True
    )
    document_id = serializers.UUIDField(source="document.id", read_only=True)
    parent = serializers.PrimaryKeyRelatedField(
        queryset=DocumentComment.objects.all(),
        required=False,
        allow_null=True,
    )
    user_id = serializers.UUIDField(source="user.id", read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)
    mentions = serializers.SerializerMethodField()

    class Meta:
        model = DocumentComment
        fields = [
            "id",
            "document",
            "document_id",
            "parent",
            "text",
            "anchor_type",
            "anchor_reference",
            "mentions",
            "user_id",
            "username",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "document_id",
            "mentions",
            "user_id",
            "username",
            "created_at",
            "updated_at",
        ]

    def get_mentions(self, obj):
        return list(
            DocumentMention.objects.filter(comment=obj).values_list(
                "mentioned_user_id", flat=True
            )
        )


class DocumentCommentResponseSerializer(DocumentCommentSerializer):
    idempotent = serializers.BooleanField(read_only=True)

    class Meta(DocumentCommentSerializer.Meta):
        fields = ["idempotent", *DocumentCommentSerializer.Meta.fields]
        read_only_fields = [
            *DocumentCommentSerializer.Meta.read_only_fields,
            "idempotent",
        ]


class DocumentNotificationSerializer(serializers.ModelSerializer):
    document_id = serializers.UUIDField(source="document.id", read_only=True)
    comment_id = serializers.IntegerField(source="comment.id", read_only=True)

    class Meta:
        model = DocumentNotification
        fields = [
            "id",
            "event_type",
            "document_id",
            "comment_id",
            "payload",
            "created_at",
            "read_at",
        ]
        read_only_fields = ["id", "created_at"]


class SavedSearchSerializer(serializers.ModelSerializer):
    class Meta:
        model = SavedSearch
        fields = [
            "id",
            "name",
            "query",
            "filters",
            "enable_alerts",
            "alert_frequency",
            "last_run_at",
            "next_run_at",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "last_run_at",
            "next_run_at",
            "created_at",
            "updated_at",
        ]


class SavedSearchResponseSerializer(SavedSearchSerializer):
    idempotent = serializers.BooleanField(read_only=True)

    class Meta(SavedSearchSerializer.Meta):
        fields = ["idempotent", *SavedSearchSerializer.Meta.fields]
        read_only_fields = [
            *SavedSearchSerializer.Meta.read_only_fields,
            "idempotent",
        ]


class DocumentNotificationStatusSerializer(serializers.Serializer):
    idempotent = serializers.BooleanField(read_only=True)
    status = serializers.CharField()


class DocumentCollectionSerializer(serializers.ModelSerializer):
    case = serializers.SerializerMethodField()

    class Meta:
        model = DocumentCollection
        fields = [
            "id",
            "collection_id",
            "key",
            "name",
            "type",
            "visibility",
            "metadata",
            "embedding_profile",
            "case",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_case(self, obj):
        case_obj = getattr(obj, "case", None)
        if case_obj is None:
            return None
        return {
            "id": str(case_obj.id),
            "external_id": case_obj.external_id,
            "title": case_obj.title,
            "status": case_obj.status,
        }
