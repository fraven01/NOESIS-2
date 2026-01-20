"""Base settings for noesis2 project."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import timedelta

import environ

from noesis2.api import schema as api_schema

from common.logging import configure_logging
from celery.schedules import crontab
from kombu import Queue


logger = logging.getLogger(__name__)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
# This file is at noesis2/settings/base.py, so project root is three parents up
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# django-environ setup
env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

configure_logging()


DEFAULT_EMBEDDING_DIMENSION = env.int("EMBEDDINGS_DIM", default=1536)
# Demo uses the same embedding dimension as standard by default
DEMO_EMBEDDING_DIMENSION = env.int(
    "DEMO_EMBEDDINGS_DIM", default=DEFAULT_EMBEDDING_DIMENSION
)

EMBEDDINGS_PROVIDER = env("EMBEDDINGS_PROVIDER", default="litellm")
EMBEDDINGS_MODEL_PRIMARY = env("EMBEDDINGS_MODEL_PRIMARY", default="oai-embed-small")
EMBEDDINGS_MODEL_FALLBACK = env("EMBEDDINGS_MODEL_FALLBACK", default="oai-embed-small")
EMBEDDINGS_MODEL_VERSION = env("EMBEDDINGS_MODEL_VERSION", default="v1")
# Demo uses the same embedding model as standard by default
DEMO_EMBEDDINGS_MODEL = env("DEMO_EMBEDDINGS_MODEL", default=EMBEDDINGS_MODEL_PRIMARY)
DEMO_EMBEDDINGS_MODEL_VERSION = env(
    "DEMO_EMBEDDINGS_MODEL_VERSION", default=EMBEDDINGS_MODEL_VERSION
)
EMBEDDINGS_BATCH_SIZE = env.int("EMBEDDINGS_BATCH_SIZE", default=64)
EMBEDDINGS_TIMEOUT_SECONDS = env.float("EMBEDDINGS_TIMEOUT_SECONDS", default=20.0)
AUTO_CREATE_CASES = env.bool("AUTO_CREATE_CASES", default=True)


if "RAG_VECTOR_STORES" not in globals():
    rag_schema_override = os.getenv("RAG_VECTOR_SCHEMA", "").strip()
    if not rag_schema_override:
        rag_schema_override = os.getenv("DEV_TENANT_SCHEMA", "").strip()
    default_vector_schema = rag_schema_override or "rag"

    RAG_VECTOR_STORES = {
        "global": {
            "backend": "pgvector",
            "schema": default_vector_schema,
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
            # "dsn_env": "RAG_DATABASE_URL",
        },
        "demo": {
            "backend": "pgvector",
            "schema": "rag_demo",
            "dimension": DEMO_EMBEDDING_DIMENSION,
        },
    }

    # Beispiel für ein Setup mit dediziertem Scope:
    # RAG_VECTOR_STORES = {
    #     "global": {
    #         "backend": "pgvector",
    #         "default": True,
    #     },
    #     "enterprise": {
    #         "backend": "pgvector",
    #         "schema": "rag_enterprise",
    #         "tenants": ["<uuid-tenant-id>"],
    #         "schemas": ["acme_prod"],
    #     },
    # }

RAG_INDEX_KIND = env("RAG_INDEX_KIND", default="HNSW").upper()
RAG_HNSW_M = env.int("RAG_HNSW_M", default=32)
RAG_HNSW_EF_CONSTRUCTION = env.int("RAG_HNSW_EF_CONSTRUCTION", default=200)
RAG_HNSW_EF_SEARCH = env.int("RAG_HNSW_EF_SEARCH", default=80)
RAG_IVF_LISTS = env.int("RAG_IVF_LISTS", default=2048)
RAG_IVF_PROBES = env.int("RAG_IVF_PROBES", default=64)
RAG_MIN_SIM = env.float("RAG_MIN_SIM", default=0.15)
RAG_TRGM_LIMIT = env.float("RAG_TRGM_LIMIT", default=0.1)
RAG_LEXICAL_MODE = env("RAG_LEXICAL_MODE", default="trgm")
RAG_HYDE_ENABLED = env.bool("RAG_HYDE_ENABLED", default=False)
RAG_HYDE_MODEL_LABEL = env("RAG_HYDE_MODEL_LABEL", default="simple-query")
RAG_HYDE_MAX_CHARS = env.int("RAG_HYDE_MAX_CHARS", default=2000)
RAG_HYBRID_ALPHA = env.float("RAG_HYBRID_ALPHA", default=0.7)
RAG_RRF_K = env.int("RAG_RRF_K", default=60)
RAG_MAX_CANDIDATES = env.int("RAG_MAX_CANDIDATES", default=200)
RAG_CANDIDATE_POLICY = env("RAG_CANDIDATE_POLICY", default="error")
RAG_CHUNK_TARGET_TOKENS = env.int("RAG_CHUNK_TARGET_TOKENS", default=450)
RAG_CHUNK_OVERLAP_TOKENS = env.int("RAG_CHUNK_OVERLAP_TOKENS", default=80)
RAG_HARD_DELETE_VACUUM = env.bool("RAG_HARD_DELETE_VACUUM", default=False)
RAG_HARD_DELETE_REINDEX_THRESHOLD = env.int(
    "RAG_HARD_DELETE_REINDEX_THRESHOLD", default=0
)

# Chunker Configuration (MODEL_ROUTING.yaml labels)
RAG_CHUNKER_MODE = env("RAG_CHUNKER_MODE", default="late")  # late | agentic | hybrid
RAG_LATE_CHUNK_MODEL = env(
    "RAG_LATE_CHUNK_MODEL", default="embedding"
)  # MODEL_ROUTING.yaml label
RAG_LATE_CHUNK_MAX_TOKENS = env.int("RAG_LATE_CHUNK_MAX_TOKENS", default=8000)
RAG_AGENTIC_CHUNK_MODEL = env(
    "RAG_AGENTIC_CHUNK_MODEL", default="agentic-chunk"
)  # MODEL_ROUTING.yaml label
RAG_ENABLE_QUALITY_METRICS = env.bool("RAG_ENABLE_QUALITY_METRICS", default=True)
RAG_QUALITY_EVAL_MODEL = env(
    "RAG_QUALITY_EVAL_MODEL", default="quality-eval"
)  # MODEL_ROUTING.yaml label

# Phase 2: SOTA Embedding-based Similarity
RAG_USE_EMBEDDING_SIMILARITY = env.bool("RAG_USE_EMBEDDING_SIMILARITY", default=False)
RAG_CHUNKING_WINDOW_SIZE = env.int("RAG_CHUNKING_WINDOW_SIZE", default=3)
RAG_CHUNKING_BATCH_SIZE = env.int("RAG_CHUNKING_BATCH_SIZE", default=16)
RAG_USE_CONTENT_BASED_IDS = env.bool("RAG_USE_CONTENT_BASED_IDS", default=True)
RAG_ADAPTIVE_CHUNKING_ENABLED = env.bool("RAG_ADAPTIVE_CHUNKING_ENABLED", default=True)
RAG_ASSET_CHUNKS_ENABLED = env.bool("RAG_ASSET_CHUNKS_ENABLED", default=True)

if "RAG_EMBEDDING_PROFILES" not in globals():
    RAG_EMBEDDING_PROFILES = {
        "standard": {
            "model": EMBEDDINGS_MODEL_PRIMARY,
            "model_version": EMBEDDINGS_MODEL_VERSION,
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
            "vector_space": "global",
            "chunk_hard_limit": 512,
        },
        "demo": {
            "model": DEMO_EMBEDDINGS_MODEL,
            "model_version": DEMO_EMBEDDINGS_MODEL_VERSION,
            "dimension": DEMO_EMBEDDING_DIMENSION,
            "vector_space": "demo",
            "chunk_hard_limit": 1024,
        },
    }

RAG_DEFAULT_EMBEDDING_PROFILE = env("RAG_DEFAULT_EMBEDDING_PROFILE", default="standard")

_default_routing_rules_path = BASE_DIR / "config" / "rag_routing_rules.yaml"
RAG_ROUTING_RULES_PATH = Path(
    env("RAG_ROUTING_RULES_PATH", default=str(_default_routing_rules_path))
)


def _load_common_headers_table() -> str:
    """Return the reference markdown table describing shared API headers."""

    reference_path = BASE_DIR / "docs" / "api" / "reference.md"
    try:
        lines: List[str] = reference_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        logger.info("Common header reference table not found at %s", reference_path)
        return ""

    try:
        start_index = next(
            index for index, line in enumerate(lines) if line.startswith("| Header |")
        )
    except StopIteration:
        return ""

    end_index = start_index
    while end_index < len(lines) and lines[end_index].startswith("|"):
        end_index += 1

    table_lines = lines[start_index:end_index]
    if not table_lines:
        return ""

    info_box_lines = ["> **Common Headers**", ">"]
    info_box_lines.extend(f"> {line}" for line in table_lines)
    return "\n".join(info_box_lines)


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

# Testing flag (auto-detected for pytest, even before PYTEST_CURRENT_TEST exists)
TESTING = bool(
    os.environ.get("PYTEST_CURRENT_TEST")
    or os.environ.get("PYTEST_XDIST_WORKER")
    or os.environ.get("PYTEST_RUNNING")
    or ("pytest" in sys.modules)
)

# PII configuration defaults
PII_MODE = os.getenv("PII_MODE", "industrial")
PII_POLICY = os.getenv("PII_POLICY", "balanced")
PII_DETERMINISTIC = os.getenv("PII_DETERMINISTIC", "true").lower() == "true"
PII_POST_RESPONSE = os.getenv("PII_POST_RESPONSE", "false").lower() == "true"
PII_LOGGING_REDACTION = os.getenv("PII_LOGGING_REDACTION", "true").lower() == "true"
PII_HMAC_SECRET = os.getenv("PII_HMAC_SECRET", "")
PII_NAME_DETECTION = os.getenv("PII_NAME_DETECTION", "false").lower() == "true"

# Guardrail observability sampling allowlist (rule identifiers)
AI_GUARDRAIL_SAMPLE_ALLOWLIST = env.list("AI_GUARDRAIL_SAMPLE_ALLOWLIST", default=[])

# Ingestion pipeline flags
INGESTION_PII_MASK_ENABLED = (
    os.getenv("INGESTION_PII_MASK_ENABLED", "true").lower() == "true"
)

ENABLE_API_DOCS = env.bool("ENABLE_API_DOCS", default=False)
API_DOCS_TITLE = env("API_DOCS_TITLE", default="NOESIS 2 API")
API_IMAGE_TAG = env("API_IMAGE_TAG", default="")
API_DOCS_VERSION_LABEL = env("API_DOCS_VERSION_LABEL", default=API_IMAGE_TAG.strip())
ENABLE_SWAGGER_TRY_IT_OUT = env.bool("ENABLE_SWAGGER_TRY_IT_OUT", default=False)
COMMON_HEADERS_INFO_BOX = _load_common_headers_table()
API_DOCS_DESCRIPTION = "OpenAPI schema for NOESIS 2 multi-tenant endpoints."
if COMMON_HEADERS_INFO_BOX:
    API_DOCS_DESCRIPTION = f"{API_DOCS_DESCRIPTION}\n\n{COMMON_HEADERS_INFO_BOX}"

# Documents repository: default to the database-backed adapter; can be overridden
# (e.g., to the filesystem adapter) via env for local development.


# Crawler HTTP adapter defaults
CRAWLER_HTTP_USER_AGENT = env("CRAWLER_HTTP_USER_AGENT", default="noesis-crawler/1.0")


# Application definition

# Apps that live in the ``public`` schema and are shared across all tenants
# Keep this minimal. Models here exist only in the public schema.
SHARED_APPS = [
    "customers.apps.CustomersConfig",
]

# In tests, include a tiny helper app that ensures pg_trgm
# is available in the public schema for the Django test DB.
if bool(os.environ.get("PYTEST_CURRENT_TEST")):
    SHARED_APPS.append("testsupport")

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
    "channels",
    "rest_framework",
    "drf_spectacular",
    "crawler",
    "cases.apps.CasesConfig",
    "documents",
    "ai_core",
    "llm_worker",
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
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "ai_core.middleware.PIISessionScopeMiddleware",
    "ai_core.middleware.RequestContextMiddleware",
    "noesis2.api.middleware.DeprecationHeaderMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "theme.middleware.SimulatedUserMiddleware",
    "profiles.middleware.ExternalAccountExpiryMiddleware",
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
ASGI_APPLICATION = "noesis2.asgi.application"


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

if TESTING:
    CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
else:
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [REDIS_URL]},
        }
    }

# Celery Configuration
# CELERY_RESULT_BACKEND is required for synchronous task result retrieval (async_result.get())
# used in the graph worker timeout pattern. Without this, async_result.get() will fail.
# Mode: Sync-wait with timeout → Returns 200 OK if task completes, 202 Accepted on timeout.
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

if TESTING:
    CELERY_TASK_ALWAYS_EAGER = True
    CELERY_TASK_EAGER_PROPAGATES_EXCEPTIONS = True

# Graph worker timeout (seconds)
# Maximum time to wait for a graph execution in the worker queue
# before returning 202 Accepted with a task_id for async polling.
# Wenn Result-Backend fehlt, fungiert Web-Pfad automatisch als 202-Fallback ohne .get()-Ergebnis.
GRAPH_WORKER_TIMEOUT_S = env.int("GRAPH_WORKER_TIMEOUT_S", default=45)

CELERY_TASK_DEFAULT_QUEUE = "default"
CELERY_TASK_DEFAULT_EXCHANGE = "default"
CELERY_TASK_DEFAULT_ROUTING_KEY = "default"

_QUEUE_MAX_PRIORITY = env.int("CELERY_QUEUE_MAX_PRIORITY", default=10)
_DLQ_TTL_MS = env.int("CELERY_DLQ_TTL_MS", default=7 * 24 * 60 * 60 * 1000)
CELERY_DLQ_CLEANUP_S = env.int("CELERY_DLQ_CLEANUP_S", default=3600)
CELERY_DLQ_ALERT_S = env.int("CELERY_DLQ_ALERT_S", default=300)
CELERY_DLQ_ALERT_THRESHOLD = env.int("CELERY_DLQ_ALERT_THRESHOLD", default=10)
CELERY_TENANT_RATE_LIMIT_ENABLED = env.bool(
    "CELERY_TENANT_RATE_LIMIT_ENABLED", default=True
)
CELERY_TASK_REJECT_ON_WORKER_LOST = True

CELERY_TASK_QUEUES = (
    Queue(
        "default",
        routing_key="default",
        queue_arguments={"x-max-priority": _QUEUE_MAX_PRIORITY},
    ),
    Queue(
        "agents-high",
        routing_key="agents-high",
        queue_arguments={"x-max-priority": _QUEUE_MAX_PRIORITY},
    ),
    Queue(
        "agents-low",
        routing_key="agents-low",
        queue_arguments={"x-max-priority": _QUEUE_MAX_PRIORITY},
    ),
    Queue(
        "ingestion",
        routing_key="ingestion",
        queue_arguments={"x-max-priority": _QUEUE_MAX_PRIORITY},
    ),
    Queue(
        "ingestion-bulk",
        routing_key="ingestion-bulk",
        queue_arguments={"x-max-priority": _QUEUE_MAX_PRIORITY},
    ),
    Queue(
        "dead_letter",
        routing_key="dead_letter",
        queue_arguments={"x-message-ttl": _DLQ_TTL_MS},
    ),
    Queue(
        "rag_delete",
        routing_key="rag_delete",
        queue_arguments={"x-max-priority": _QUEUE_MAX_PRIORITY},
    ),
)

CELERY_TASK_ROUTES = ("common.celery.route_task",)

CELERY_BEAT_SCHEDULE = {
    "dlq-cleanup": {
        "task": "ai_core.tasks.cleanup_dead_letter",
        "schedule": timedelta(seconds=CELERY_DLQ_CLEANUP_S),
        "options": {"queue": "default"},
    },
    "dlq-alert": {
        "task": "ai_core.tasks.alert_dead_letter",
        "schedule": timedelta(seconds=CELERY_DLQ_ALERT_S),
        "options": {"queue": "default"},
    },
    "rag-embedding-drift": {
        "task": "ai_core.tasks.embedding_drift_check",
        "schedule": crontab(day_of_month=1, hour=2, minute=0),
        "options": {"queue": "default"},
    },
    "document-version-cleanup": {
        "task": "documents.tasks.cleanup_document_versions",
        "schedule": crontab(hour=3, minute=0),
        "options": {"queue": "default"},
    },
    "rag-rerank-weight-update": {
        "task": "ai_core.tasks.rag_feedback.update_weights",
        "schedule": timedelta(hours=6),
        "options": {"queue": "default"},
    },
}


ADMINS = [
    (
        env("ADMIN_NAME", default="Admin"),
        env("ADMIN_EMAIL", default="admin@example.com"),
    )
]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
DOCUMENT_COMMENT_CREATE_RATE = env("DOCUMENT_COMMENT_CREATE_RATE", default="60/min")

# Logging / observability
LOGGING_CONFIG = "common.logging.configure_django_logging"

LOGGING_ALLOW_UNMASKED_CONTEXT = env.bool(
    "LOGGING_ALLOW_UNMASKED_CONTEXT", default=False
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
        },
        "json_console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
        },
        "mail_admins": {
            "level": "ERROR",
            "class": "django.utils.log.AdminEmailHandler",
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_AUTHENTICATION_CLASSES": [
        # Policy-enforcing authentication classes that check:
        # - Account expiry (EXTERNAL accounts)
        # - Active status (user.is_active, profile.is_active)
        # - (Future) Concurrent sessions, IP restrictions, etc.
        "profiles.authentication.PolicyEnforcingSessionAuthentication",
        "profiles.authentication.PolicyEnforcingBasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
}

TENANT_HEADER_COMPONENTS = api_schema.tenant_header_components()
SWAGGER_SUBMIT_METHODS = [
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "options",
    "head",
    "trace",
]


def build_swagger_ui_settings(enable_try_it_out: bool) -> Dict[str, object]:
    supported_methods = SWAGGER_SUBMIT_METHODS if enable_try_it_out else []
    return {
        "tryItOutEnabled": enable_try_it_out,
        "supportedSubmitMethods": supported_methods,
        "defaultModelRendering": "example",
        "docExpansion": "none",
        "displayRequestDuration": True,
        "persistAuthorization": True,
    }


SPECTACULAR_SETTINGS = {
    "TITLE": API_DOCS_TITLE,
    "DESCRIPTION": API_DOCS_DESCRIPTION,
    "VERSION": "v1",
    "SECURITY": [{api_schema.ADMIN_BEARER_AUTH_SCHEME: []}],
    "DISABLE_ERRORS_AND_WARNINGS": True,
    "APPEND_COMPONENTS": {
        "parameters": TENANT_HEADER_COMPONENTS,
        "headers": {
            api_schema.TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME: api_schema.TRACE_ID_RESPONSE_HEADER_COMPONENT,
        },
        "securitySchemes": {
            api_schema.ADMIN_BEARER_AUTH_SCHEME: {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "API Key",
                "description": "LiteLLM Admin bearer token used for privileged endpoints.",
            },
            "cookieAuth": {
                "type": "apiKey",
                "in": "cookie",
                "name": "sessionid",
                "description": "Django session cookie with account policy enforcement.",
            },
            "basicAuth": {
                "type": "http",
                "scheme": "basic",
                "description": "HTTP Basic authentication with account policy enforcement.",
            },
        },
    },
    "POSTPROCESSING_HOOKS": [
        "noesis2.api.schema.normalize_openapi_examples",
        "noesis2.api.schema.inject_trace_response_header",
        "drf_spectacular.hooks.postprocess_schema_enums",
    ],
    "SWAGGER_UI_SETTINGS": build_swagger_ui_settings(ENABLE_SWAGGER_TRY_IT_OUT),
    "REDOC_UI_SETTINGS": {
        "hideDownloadButton": False,
    },
}

API_DEPRECATIONS = {
    "ai-core-legacy": {
        "deprecation": "Wed, 01 Jan 2025 00:00:00 GMT",
        "sunset": "Tue, 01 Jul 2025 00:00:00 GMT",
    }
}

# LiteLLM / AI Core
LITELLM_BASE_URL = env("LITELLM_BASE_URL", default="http://localhost:4000")
LITELLM_MASTER_KEY = env("LITELLM_MASTER_KEY", default="")
DEPLOYMENT_ENV = env("DEPLOYMENT_ENV", default="development").lower()
LANGFUSE_HOST = env("LANGFUSE_HOST", default="http://localhost:3100")
LANGFUSE_PUBLIC_KEY = env("LANGFUSE_PUBLIC_KEY", default="")
LANGFUSE_SECRET_KEY = env("LANGFUSE_SECRET_KEY", default="")
LANGFUSE_SAMPLE_RATE = float(env("LANGFUSE_SAMPLE_RATE", default=1.0))
AI_CORE_RATE_LIMIT_QUOTA = int(env("AI_CORE_RATE_LIMIT_QUOTA", default=60))
AI_CORE_RATE_LIMIT_AGENTS_QUOTA = int(
    env("AI_CORE_RATE_LIMIT_AGENTS_QUOTA", default=100)
)
AI_CORE_RATE_LIMIT_INGESTION_QUOTA = int(
    env("AI_CORE_RATE_LIMIT_INGESTION_QUOTA", default=500)
)
RERANK_MODEL_PRESET = env("RERANK_MODEL_PRESET", default="fast")
RERANK_CACHE_TTL_SECONDS = int(env("RERANK_CACHE_TTL_SECONDS", default=900))
LITELLM_CIRCUIT_FAILURE_THRESHOLD = env.int(
    "LITELLM_CIRCUIT_FAILURE_THRESHOLD", default=5
)
LITELLM_CIRCUIT_BASE_BACKOFF_S = env.float(
    "LITELLM_CIRCUIT_BASE_BACKOFF_S", default=60.0
)
LITELLM_CIRCUIT_MAX_BACKOFF_S = env.float(
    "LITELLM_CIRCUIT_MAX_BACKOFF_S", default=600.0
)

# Google Custom Search (for web_acquisition_graph and collection_search)
GOOGLE_CUSTOM_SEARCH_API_KEY = env("GOOGLE_CUSTOM_SEARCH_API_KEY", default=None)
_search_engine_id = env("GOOGLE_CUSTOM_SEARCH_ENGINE_ID", default=None)
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = (
    _search_engine_id.replace("cx=", "") if _search_engine_id else None
)

# Guard: prevent accidental use of production Langfuse keys in non-prod environments
if DEPLOYMENT_ENV in {"development", "dev", "staging", "test", "testing"}:
    lf_pk = (LANGFUSE_PUBLIC_KEY or "").lower()
    lf_sk = (LANGFUSE_SECRET_KEY or "").lower()
    suspicious_tokens = ("prod", "pk_prod_", "sk_prod_", "pk-live", "sk-live", "live_")
    if any(tok in lf_pk for tok in suspicious_tokens) or any(
        tok in lf_sk for tok in suspicious_tokens
    ):
        raise RuntimeError(
            "Refusing to start with production Langfuse keys in non-production environment. "
            "Set DEPLOYMENT_ENV=production explicitly if this is intentional, or swap keys to dev/staging."
        )
