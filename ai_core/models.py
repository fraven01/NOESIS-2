from __future__ import annotations

from django.db import models


class RagFeedbackEvent(models.Model):
    """Store implicit feedback events for rerank weight learning."""

    FEEDBACK_USED_SOURCE = "used_source"
    FEEDBACK_CLICK = "click"

    FEEDBACK_CHOICES = [
        (FEEDBACK_USED_SOURCE, "Used Source"),
        (FEEDBACK_CLICK, "Click"),
    ]

    tenant_id = models.CharField(max_length=128)
    case_id = models.CharField(max_length=128, null=True, blank=True)
    collection_id = models.CharField(max_length=128, null=True, blank=True)
    workflow_id = models.CharField(max_length=128, null=True, blank=True)
    thread_id = models.CharField(max_length=128, null=True, blank=True)
    trace_id = models.CharField(max_length=128, null=True, blank=True)
    quality_mode = models.CharField(max_length=64, default="standard")
    feedback_type = models.CharField(max_length=32, choices=FEEDBACK_CHOICES)
    query_text = models.TextField(null=True, blank=True)
    source_id = models.CharField(max_length=256, null=True, blank=True)
    source_label = models.CharField(max_length=256, null=True, blank=True)
    document_id = models.CharField(max_length=128, null=True, blank=True)
    chunk_id = models.CharField(max_length=256, null=True, blank=True)
    relevance_score = models.FloatField(null=True, blank=True)
    feature_payload = models.JSONField(null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant_id", "quality_mode", "created_at"]),
            models.Index(fields=["tenant_id", "feedback_type", "created_at"]),
        ]


class RagRerankWeight(models.Model):
    """Store learned rerank feature weights per tenant/quality mode."""

    tenant_id = models.CharField(max_length=128)
    quality_mode = models.CharField(max_length=64, default="standard")
    weights = models.JSONField()
    sample_count = models.IntegerField(default=0)
    source = models.CharField(max_length=64, default="learned")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("tenant_id", "quality_mode")
        indexes = [
            models.Index(fields=["tenant_id", "quality_mode"]),
        ]


__all__ = ["RagFeedbackEvent", "RagRerankWeight"]
