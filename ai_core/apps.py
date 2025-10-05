from django.apps import AppConfig

from .rag.embedding_config import validate_embedding_configuration
from .rag.routing_rules import validate_routing_rules
from .rag.vector_schema import validate_vector_schemas


class AiCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_core"

    def ready(self) -> None:
        """Perform application bootstrap when the app registry is ready."""

        super().ready()

        validate_embedding_configuration()
        validate_routing_rules()
        validate_vector_schemas()

        from .graph.bootstrap import bootstrap

        bootstrap()
