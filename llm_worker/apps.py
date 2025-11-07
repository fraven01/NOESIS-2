from django.apps import AppConfig


class LlmWorkerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "llm_worker"
    verbose_name = "LLM Worker"
