"""Models for AI core functionality."""

from django.db import models

from common.models import TimestampedModel


class SearchProviderConfiguration(TimestampedModel):
    """Tenant-specific configuration for web search providers.

    This model allows each tenant to configure their own search provider
    (e.g., Google Custom Search, Bing, Brave) with provider-specific settings.
    """

    class ProviderType(models.TextChoices):
        GOOGLE_CSE = "google_cse", "Google Custom Search"
        BING_SEARCH = "bing_search", "Bing Search"
        BRAVE_SEARCH = "brave_search", "Brave Search"

    tenant = models.ForeignKey(
        "customers.Tenant",
        on_delete=models.CASCADE,
        related_name="search_provider_configs",
        help_text="The tenant this configuration belongs to",
    )
    provider_type = models.CharField(
        max_length=32,
        choices=ProviderType.choices,
        help_text="The type of search provider (e.g., google_cse, bing_search)",
    )
    configuration = models.JSONField(
        default=dict,
        help_text="Provider-specific configuration (e.g., api_key, search_engine_id)",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this configuration is active and should be used",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "provider_type"],
                condition=models.Q(is_active=True),
                name="unique_active_provider_per_tenant",
            )
        ]
        indexes = [
            models.Index(fields=["tenant", "is_active"], name="search_prov_tenant_idx"),
        ]
        verbose_name = "Search Provider Configuration"
        verbose_name_plural = "Search Provider Configurations"

    def __str__(self) -> str:
        return f"{self.tenant} - {self.get_provider_type_display()} ({'Active' if self.is_active else 'Inactive'})"
