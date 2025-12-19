import sys
import django
from django.conf import settings


def run_tests():
    # 1. Configure Django sparsely (no DB connection needed if mocks hold)
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="test-secret",
            INSTALLED_APPS=[
                "django.contrib.contenttypes",
                "django.contrib.auth",
                "ai_core",
                "documents",
                "customers",
                "cases",
            ],
            DATABASES={
                "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}
            },
            # Add other required settings if tests complain
            TENANT_MODEL="customers.Tenant",
            TENANT_DOMAIN_MODEL="customers.Domain",
            CRAWLER_DEFAULT_WORKFLOW_ID="wf-1",
        )
        django.setup()

    # 2. Run Pytest
    import pytest

    sys.exit(
        pytest.main(["ai_core/tests/integration/test_universal_migration.py", "-v"])
    )


if __name__ == "__main__":
    run_tests()
