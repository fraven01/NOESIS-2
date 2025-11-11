from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("common", "0002_domainpolicyoverride"),
    ]

    operations = [
        migrations.CreateModel(
            name="DomainPolicy",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("tenant_id", models.CharField(max_length=255)),
                ("domain", models.CharField(max_length=512)),
                (
                    "action",
                    models.CharField(
                        choices=[("boost", "Boost"), ("reject", "Reject")],
                        default="boost",
                        max_length=16,
                    ),
                ),
                ("priority", models.PositiveIntegerField(default=0)),
            ],
            options={
                "indexes": [
                    models.Index(fields=("tenant_id", "action"), name="domain_policy_action_idx"),
                ],
            },
        ),
        migrations.AddConstraint(
            model_name="domainpolicy",
            constraint=models.UniqueConstraint(
                fields=("tenant_id", "domain"), name="domain_policy_unique_rule"
            ),
        ),
    ]

