from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0022_create_external_notifications_phase4b"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="audit_meta",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="frameworkprofile",
            name="audit_meta",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
