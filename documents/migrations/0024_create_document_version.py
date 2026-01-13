from __future__ import annotations

import uuid

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0023_add_audit_meta_fields"),
    ]

    operations = [
        migrations.CreateModel(
            name="DocumentVersion",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        primary_key=True,
                        default=uuid.uuid4,
                        editable=False,
                        serialize=False,
                    ),
                ),
                (
                    "version_label",
                    models.CharField(max_length=64, null=True, blank=True),
                ),
                ("sequence", models.IntegerField()),
                ("label_sequence", models.IntegerField()),
                ("is_latest", models.BooleanField(default=True)),
                ("deleted_at", models.DateTimeField(null=True, blank=True)),
                ("normalized_document", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "created_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="created_document_versions",
                        to="users.user",
                    ),
                ),
                (
                    "created_by_service_id",
                    models.CharField(max_length=100, null=True, blank=True),
                ),
                (
                    "document",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="versions",
                        to="documents.document",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=("document", "is_latest"),
                        name="document_version_latest_idx",
                    ),
                    models.Index(
                        fields=("document", "created_at"),
                        name="document_version_created_idx",
                    ),
                    models.Index(
                        fields=("document", "version_label", "label_sequence"),
                        name="document_version_label_idx",
                    ),
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("document", "sequence"),
                        name="document_version_unique_sequence",
                    ),
                ],
            },
        ),
    ]
