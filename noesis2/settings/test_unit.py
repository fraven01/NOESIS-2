from .base import *  # noqa: F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Remove django_tenants from INSTALLED_APPS and MIDDLEWARE
INSTALLED_APPS = [
    app for app in INSTALLED_APPS if app != "django_tenants"  # noqa: F405
]
MIDDLEWARE = [mw for mw in MIDDLEWARE if "django_tenants" not in mw]  # noqa: F405
DATABASE_ROUTERS = []

# Disable TenantModel
TENANT_MODEL = None
TENANT_DOMAIN_MODEL = None
