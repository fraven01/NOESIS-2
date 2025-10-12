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
        child=serializers.JSONField(),
        required=False,
        help_text="Arbitrary metadata attached to the workflow state.",
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
    tenant_id = serializers.CharField()
    case_id = serializers.CharField()


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


class RagSnippetSerializer(serializers.Serializer):
    """Serializer describing individual retrieval snippets."""

    id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    text = serializers.CharField()
    score = serializers.FloatField()
    source = serializers.CharField(allow_blank=True)
    hash = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    meta = serializers.JSONField(required=False)


class RagRetrievalRoutingSerializer(serializers.Serializer):
    """Routing metadata emitted by the retrieval step."""

    profile = serializers.CharField()
    vector_space_id = serializers.CharField()


class RagRetrievalMetaSerializer(serializers.Serializer):
    """Serializer for the retrieval diagnostics payload."""

    alpha = serializers.FloatField()
    min_sim = serializers.FloatField()
    top_k_effective = serializers.IntegerField()
    matches_returned = serializers.IntegerField()
    max_candidates_effective = serializers.IntegerField()
    vector_candidates = serializers.IntegerField()
    lexical_candidates = serializers.IntegerField()
    deleted_matches_blocked = serializers.IntegerField()
    visibility_effective = serializers.CharField()
    took_ms = serializers.IntegerField()
    routing = RagRetrievalRoutingSerializer()


class RagQueryResponseSerializer(serializers.Serializer):
    """Response payload returned by the production RAG query endpoint.

    MVP 2025-10 — Breaking Contract v2: Response enthält answer, prompt_version, retrieval, snippets.
    """

    answer = serializers.CharField(allow_blank=True)
    prompt_version = serializers.CharField()
    retrieval = RagRetrievalMetaSerializer()
    snippets = RagSnippetSerializer(many=True)
    diagnostics = serializers.DictField(
        child=serializers.JSONField(), required=False, help_text="Non-contract extras."
    )


class TenantDemoResponseSerializer(serializers.Serializer):
    """Response body returned by the tenant demo view."""

    status = serializers.CharField()
