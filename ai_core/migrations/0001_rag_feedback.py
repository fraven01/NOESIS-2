from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="RagFeedbackEvent",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("tenant_id", models.CharField(max_length=128)),
                ("case_id", models.CharField(blank=True, max_length=128, null=True)),
                (
                    "collection_id",
                    models.CharField(blank=True, max_length=128, null=True),
                ),
                (
                    "workflow_id",
                    models.CharField(blank=True, max_length=128, null=True),
                ),
                ("thread_id", models.CharField(blank=True, max_length=128, null=True)),
                ("trace_id", models.CharField(blank=True, max_length=128, null=True)),
                ("quality_mode", models.CharField(default="standard", max_length=64)),
                (
                    "feedback_type",
                    models.CharField(
                        choices=[("used_source", "Used Source"), ("click", "Click")],
                        max_length=32,
                    ),
                ),
                ("query_text", models.TextField(blank=True, null=True)),
                ("source_id", models.CharField(blank=True, max_length=256, null=True)),
                (
                    "source_label",
                    models.CharField(blank=True, max_length=256, null=True),
                ),
                (
                    "document_id",
                    models.CharField(blank=True, max_length=128, null=True),
                ),
                ("chunk_id", models.CharField(blank=True, max_length=256, null=True)),
                ("relevance_score", models.FloatField(blank=True, null=True)),
                ("feature_payload", models.JSONField(blank=True, null=True)),
                ("metadata", models.JSONField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["tenant_id", "quality_mode", "created_at"],
                        name="ai_core_ra_tenant__5e9a6a_idx",
                    ),
                    models.Index(
                        fields=["tenant_id", "feedback_type", "created_at"],
                        name="ai_core_ra_tenant__df7d31_idx",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="RagRerankWeight",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("tenant_id", models.CharField(max_length=128)),
                ("quality_mode", models.CharField(default="standard", max_length=64)),
                ("weights", models.JSONField()),
                ("sample_count", models.IntegerField(default=0)),
                ("source", models.CharField(default="learned", max_length=64)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["tenant_id", "quality_mode"],
                        name="ai_core_ra_tenant__47a31f_idx",
                    ),
                ],
                "unique_together": {("tenant_id", "quality_mode")},
            },
        ),
    ]
