"""Django admin configuration for AI core models."""

from django.contrib import admin

from ai_core.models import SearchProviderConfiguration


@admin.register(SearchProviderConfiguration)
class SearchProviderConfigurationAdmin(admin.ModelAdmin):
    """Admin interface for SearchProviderConfiguration."""

    list_display = (
        "tenant",
        "provider_type",
        "is_active",
        "created_at",
        "updated_at",
    )
    list_filter = ("provider_type", "is_active", "created_at")
    search_fields = ("tenant__name", "tenant__schema_name")
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        (
            None,
            {
                "fields": ("tenant", "provider_type", "is_active"),
            },
        ),
        (
            "Configuration",
            {
                "fields": ("configuration",),
                "description": (
                    "Provider-specific configuration in JSON format. "
                    "For Google CSE: {'api_key': '...', 'search_engine_id': '...'}"
                ),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related("tenant")
