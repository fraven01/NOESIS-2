from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("common", "0001_drop_legacy_workflows"),
    ]

    operations = [
        migrations.CreateModel(
            name="DomainPolicyOverride",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("tenant_id", models.CharField(max_length=255, unique=True)),
                ("preferred_hosts", models.JSONField(blank=True, default=list)),
                ("blocked_hosts", models.JSONField(blank=True, default=list)),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.AddIndex(
            model_name="domainpolicyoverride",
            index=models.Index(fields=("tenant_id",), name="domain_policy_override_tenant_idx"),
        ),
    ]
