from .base import *  # noqa: F403

# Production overrides
DEBUG = False

# Static files collection target
STATIC_ROOT = BASE_DIR / "staticfiles"  # noqa: F405

# Security settings
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_SSL_REDIRECT = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "strict-origin"
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Use JSON logging in production
LOGGING["root"]["handlers"] = ["json_console"]  # noqa: F405
LOGGING["loggers"]["django.request"] = {  # noqa: F405
    "handlers": ["mail_admins", "json_console"],
    "level": "ERROR",
    "propagate": False,
}
