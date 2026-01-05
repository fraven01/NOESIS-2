from .base import *  # noqa: F403

# Enable TESTING flag for test-specific behavior (e.g., RAG tools access)
TESTING = True

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Remove django_tenants and customers from INSTALLED_APPS
INSTALLED_APPS = [
    app
    for app in INSTALLED_APPS  # noqa: F405
    if app not in ("django_tenants", "customers")  # noqa: F405
]
MIDDLEWARE = [mw for mw in MIDDLEWARE if "django_tenants" not in mw]  # noqa: F405
DATABASE_ROUTERS = []

# Disable TenantModel with a dummy valid model to satisfy imports
TENANT_MODEL = "auth.User"
TENANT_DOMAIN_MODEL = "auth.User"
