"""Django application configuration for the crawler app."""

from django.apps import AppConfig


class CrawlerConfig(AppConfig):
    """Register the crawler app containing canonicalization contracts."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "crawler"
    verbose_name = "Crawler Contracts"
