from .base import *  # noqa: F403

# Production overrides
DEBUG = False

# Static files collection target
STATIC_ROOT = BASE_DIR / "staticfiles"  # noqa: F405

# Use JSON logging in production
LOGGING["root"]["handlers"] = ["json_console"]  # noqa: F405
LOGGING["loggers"]["django.request"] = {  # noqa: F405
    "handlers": ["mail_admins", "json_console"],
    "level": "ERROR",
    "propagate": False,
}
