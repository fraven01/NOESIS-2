"""
Base settings for noesis2 project.

Split into base/development/production. Development and production import * from here.
"""

import os
from pathlib import Path
import copy
import environ
from django.utils.log import DEFAULT_LOGGING
from django.core.exceptions import ImproperlyConfigured

# Build paths inside the project like this: BASE_DIR / 'subdir'.
# This file is at noesis2/settings/base.py, so project root is three parents up
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# django-environ setup
env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env("SECRET_KEY")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = env.list(
    "ALLOWED_HOSTS",
    default=["example.com", "localhost", ".localhost", "testserver"],
)

# Testing flag (auto-detected for pytest)
TESTING = bool(os.environ.get("PYTEST_CURRENT_TEST"))


# Application definition

# Apps that live in the ``public`` schema and are shared across all tenants
# Keep this minimal. Models here exist only in the public schema.
SHARED_APPS = [
    "customers.apps.CustomersConfig",
]

# Apps that are installed separately for each tenant schema
# Order matters for template overrides: keep `theme` before `django.contrib.admin`.
TENANT_APPS = [
    "theme.apps.ThemeConfig",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "projects",
    "documents",
    "workflows",
    "ai_core",
    "users",
    "profiles",
    "organizations",
    "common",
]

# Final installed apps: ``django_tenants`` plus shared and tenant apps
# Ensure our apps (e.g., customers) are loaded before django_tenants so our
# management commands like `create_tenant` take precedence in tests.
INSTALLED_APPS = ["django_tenants", *SHARED_APPS, *TENANT_APPS]

MIDDLEWARE = [
    "django_tenants.middleware.main.TenantMainMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "common.middleware.HeaderTenantRoutingMiddleware",
    "common.middleware.TenantSchemaMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "noesis2.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "noesis2.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases
try:
    # Try to load the PostgreSQL driver
    import psycopg2  # noqa: F401

    # Configure PostgreSQL directly from individual environment variables.
    # This avoids issues with special characters in passwords breaking a DATABASE_URL.
    DATABASES = {
        "default": {
            "ENGINE": "django_tenants.postgresql_backend",
            "NAME": env("DB_NAME"),
            "USER": env("DB_USER"),
            "PASSWORD": env("DB_PASSWORD"),
            "HOST": env("DB_HOST", default="localhost"),
            "PORT": env("DB_PORT", default="5432"),
        }
    }
    # Special configuration for Google Cloud Run: use Cloud SQL Unix socket
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        DATABASES["default"][
            "HOST"
        ] = f"/cloudsql/{os.getenv('CLOUD_SQL_CONNECTION_NAME')}"
except (ImportError, ImproperlyConfigured, Exception):
    # If the driver is not installed or env vars are missing, fall back to SQLite
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# Tenant settings
PUBLIC_SCHEMA_NAME = "public"
DATABASE_ROUTERS = ["django_tenants.routers.TenantSyncRouter"]
TENANT_MODEL = "customers.Tenant"
TENANT_DOMAIN_MODEL = "customers.Domain"


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Custom user model
AUTH_USER_MODEL = "users.User"

CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="redis://localhost:6379/0")


ADMINS = [
    (
        env("ADMIN_NAME", default="Admin"),
        env("ADMIN_EMAIL", default="admin@example.com"),
    )
]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

LOGGING = copy.deepcopy(DEFAULT_LOGGING)
LOGGING["formatters"]["verbose"] = {
    "format": "[%(asctime)s] %(levelname)s %(module)s %(message)s",
}
LOGGING["formatters"]["json"] = {
    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
}
LOGGING["handlers"]["json_console"] = {
    "class": "logging.StreamHandler",
    "formatter": "json",
}
LOGGING["handlers"]["console"]["formatter"] = "verbose"
LOGGING["root"] = {
    "handlers": ["console"],
    "level": "INFO",
}
