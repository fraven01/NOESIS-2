from .base import *  # noqa: F403

# Explicit imports to satisfy linters for names used below
from .base import SPECTACULAR_SETTINGS, build_swagger_ui_settings

# Development overrides
DEBUG = True
ALLOWED_HOSTS = ["*"]
ENABLE_API_DOCS = True
ENABLE_SWAGGER_TRY_IT_OUT = True
SPECTACULAR_SETTINGS["SWAGGER_UI_SETTINGS"] = build_swagger_ui_settings(
    True
)  # noqa: F405

# Use a more standard User-Agent for development to avoid blocking by sites like Wikipedia
CRAWLER_HTTP_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
