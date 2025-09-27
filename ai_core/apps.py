from django.apps import AppConfig


class AiCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_core"

    def ready(self) -> None:
        """Perform application bootstrap when the app registry is ready."""

        super().ready()

        from .graph.bootstrap import bootstrap

        bootstrap()
