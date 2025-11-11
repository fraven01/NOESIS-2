"""Migration for SearchProviderConfiguration model."""

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("customers", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="SearchProviderConfiguration",
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
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "provider_type",
                    models.CharField(
                        choices=[
                            ("google_cse", "Google Custom Search"),
                            ("bing_search", "Bing Search"),
                            ("brave_search", "Brave Search"),
                        ],
                        help_text="The type of search provider (e.g., google_cse, bing_search)",
                        max_length=32,
                    ),
                ),
                (
                    "configuration",
                    models.JSONField(
                        default=dict,
                        help_text="Provider-specific configuration (e.g., api_key, search_engine_id)",
                    ),
                ),
                (
                    "is_active",
                    models.BooleanField(
                        default=True,
                        help_text="Whether this configuration is active and should be used",
                    ),
                ),
                (
                    "tenant",
                    models.ForeignKey(
                        help_text="The tenant this configuration belongs to",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="search_provider_configs",
                        to="customers.tenant",
                    ),
                ),
            ],
            options={
                "verbose_name": "Search Provider Configuration",
                "verbose_name_plural": "Search Provider Configurations",
                "indexes": [
                    models.Index(
                        fields=["tenant", "is_active"], name="search_prov_tenant_idx"
                    ),
                ],
            },
        ),
        migrations.AddConstraint(
            model_name="searchproviderconfiguration",
            constraint=models.UniqueConstraint(
                condition=models.Q(("is_active", True)),
                fields=("tenant", "provider_type"),
                name="unique_active_provider_per_tenant",
            ),
        ),
    ]
