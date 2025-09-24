from __future__ import annotations

from rest_framework import serializers


class PingResponseSerializer(serializers.Serializer):
    """Response returned by the `/ai/ping/` endpoint."""

    ok = serializers.BooleanField()


class IdempotentResponseSerializer(serializers.Serializer):
    """Base serializer exposing the idempotent replay flag used by POST endpoints."""

    idempotent = serializers.BooleanField(
        required=False,
        help_text="Signals whether this payload was returned from an idempotent replay.",
    )


class IntakeRequestSerializer(serializers.Serializer):
    """Request body accepted by the agent intake endpoint."""

    prompt = serializers.CharField(required=False, allow_blank=True)
    metadata = serializers.DictField(
        child=serializers.JSONField(), required=False, help_text="Arbitrary metadata attached to the workflow state."
    )
    scope = serializers.CharField(required=False, allow_blank=True)
    needs_input = serializers.ListField(
        child=serializers.JSONField(),
        required=False,
        help_text="Optional list of user supplied needs or tasks.",
    )


class IntakeResponseSerializer(IdempotentResponseSerializer):
    """Successful response returned by the agent intake endpoint."""

    received = serializers.BooleanField()
    tenant = serializers.CharField()
    case = serializers.CharField()


class ScopeResponseSerializer(IdempotentResponseSerializer):
    """Response payload produced by the scope validation step."""

    missing = serializers.ListField(child=serializers.CharField())


class NeedsResponseSerializer(IdempotentResponseSerializer):
    """Response payload produced by the needs mapping step."""

    missing = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Entries that must be provided before needs mapping can succeed.",
    )
    mapped = serializers.BooleanField(required=False)


class SysDescResponseSerializer(IdempotentResponseSerializer):
    """Response payload produced by the system description step."""

    description = serializers.CharField(required=False)
    skipped = serializers.BooleanField(required=False)
    missing = serializers.ListField(child=serializers.CharField(), required=False)


class TenantDemoResponseSerializer(serializers.Serializer):
    """Response body returned by the tenant demo view."""

    status = serializers.CharField()
