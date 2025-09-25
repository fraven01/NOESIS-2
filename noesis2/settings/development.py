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
