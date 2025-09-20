"""
Base settings for noesis2 project.

Split into base/development/production. Development and production import * from here.
"""

import os
from pathlib import Path
import copy
import environ
from django.utils.log import DEFAULT_LOGGING

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

# Feature switches
RAG_ENABLED = env.bool("RAG_ENABLED", default=False)


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
# Note: Django resolves management command overrides in reverse INSTALLED_APPS order.
# Keeping ``django_tenants`` first and our apps (e.g., ``customers``) after it
# ensures our commands (like ``create_tenant``) override built-ins where present.
INSTALLED_APPS = ["django_tenants", *SHARED_APPS, *TENANT_APPS]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django_tenants.middleware.main.TenantMainMiddleware",
    "common.middleware.HeaderTenantRoutingMiddleware",
    "common.middleware.TenantSchemaMiddleware",
    "common.middleware.RequestLogContextMiddleware",
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
DATABASES = {"default": env.db("DATABASE_URL")}
DATABASES["default"]["ENGINE"] = "django_tenants.postgresql_backend"

# Special configuration for Google Cloud Run: use Cloud SQL Unix socket
if os.getenv("GOOGLE_CLOUD_PROJECT"):
    DATABASES["default"]["HOST"] = f"/cloudsql/{os.getenv('CLOUD_SQL_CONNECTION_NAME')}"

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

REDIS_URL = env.str("REDIS_URL")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL


ADMINS = [
    (
        env("ADMIN_NAME", default="Admin"),
        env("ADMIN_EMAIL", default="admin@example.com"),
    )
]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

# Logging / observability
LOGGING_ALLOW_UNMASKED_CONTEXT = env.bool(
    "LOGGING_ALLOW_UNMASKED_CONTEXT", default=False
)

LOGGING = copy.deepcopy(DEFAULT_LOGGING)
LOGGING["formatters"]["verbose"] = {
    "format": (
        "[%(asctime)s] %(levelname)s %(name)s "
        "trace=%(trace_id)s case=%(case_id)s "
        "tenant=%(tenant)s key_alias=%(key_alias)s %(message)s"
    ),
}

# LiteLLM / AI Core
LITELLM_BASE_URL = env("LITELLM_BASE_URL", default="http://localhost:4000")
LITELLM_MASTER_KEY = env("LITELLM_MASTER_KEY", default="")
LANGFUSE_PUBLIC_KEY = env("LANGFUSE_PUBLIC_KEY", default="")
LANGFUSE_SECRET_KEY = env("LANGFUSE_SECRET_KEY", default="")
AI_CORE_RATE_LIMIT_QUOTA = int(env("AI_CORE_RATE_LIMIT_QUOTA", default=60))
LOGGING["formatters"]["json"] = {
    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
    "fmt": (
        "%(asctime)s %(levelname)s %(name)s %(trace_id)s %(case_id)s %(tenant)s "
        "%(key_alias)s %(message)s"
    ),
}
LOGGING.setdefault("filters", {})
LOGGING["filters"]["request_task_context"] = {
    "()": "common.logging.RequestTaskContextFilter",
}
LOGGING["handlers"]["json_console"] = {
    "class": "logging.StreamHandler",
    "formatter": "json",
}
LOGGING["handlers"]["json_console"].setdefault("filters", []).append(
    "request_task_context"
)
LOGGING["handlers"]["console"].setdefault("filters", []).append(
    "request_task_context"
)
LOGGING["handlers"]["console"]["formatter"] = "verbose"
LOGGING["root"] = {
    "handlers": ["console"],
    "level": "INFO",
}
LOGGING.setdefault("loggers", {})
LOGGING["loggers"].setdefault(
    "celery",
    {
        "handlers": ["console"],
        "level": "INFO",
        "propagate": False,
    },
)
