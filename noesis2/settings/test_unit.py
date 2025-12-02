from .base import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Remove django_tenants from INSTALLED_APPS and MIDDLEWARE
INSTALLED_APPS = [app for app in INSTALLED_APPS if app != "django_tenants"]
MIDDLEWARE = [mw for mw in MIDDLEWARE if "django_tenants" not in mw]
DATABASE_ROUTERS = []

# Disable TenantModel
TENANT_MODEL = None
TENANT_DOMAIN_MODEL = None
