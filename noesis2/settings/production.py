from .base import *  # noqa: F403

# Production overrides
DEBUG = False

# Static files collection target
STATIC_ROOT = BASE_DIR / "staticfiles"  # noqa: F405

# Allow all hostnames on Google Cloud Run, as the URL is dynamically generated.
ALLOWED_HOSTS = [".run.app"]

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

# Behind Cloud Run's proxy, trust X-Forwarded-Proto for HTTPS detection
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Allow browser POSTs from Cloud Run URLs (CSRF protection)
CSRF_TRUSTED_ORIGINS = ["https://*.run.app"]

# Use JSON logging in production
handlers = LOGGING.setdefault("handlers", {})  # noqa: F405
handlers.setdefault(
    "json_console",
    {
        "class": "logging.StreamHandler",
        "level": "INFO",
    },
)

root_logger_config = LOGGING.setdefault("root", {})  # noqa: F405
root_logger_config["handlers"] = ["json_console"]
root_logger_config.setdefault("level", "INFO")

loggers = LOGGING.setdefault("loggers", {})  # noqa: F405
loggers["django.request"] = {
    "handlers": ["mail_admins", "json_console"],
    "level": "ERROR",
    "propagate": False,
}
