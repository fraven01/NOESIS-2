import sys
import subprocess
import os


def check_imports():
    print("Verifying import contracts...")

    code = """
import os
import sys
import django
from django.conf import settings

# Configure Django settings manually to avoid module loading issues
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="test_secret_key",
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "ai_core",
            "documents",
            "customers",
            "cases",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        TENANT_MODEL="customers.Tenant",
        TENANT_DOMAIN_MODEL="customers.Domain",
    )
    django.setup()

import ai_core.ingestion
import documents.parsers
import documents.parsers_pdf
import ai_core.graph.bootstrap

# Verification: Bootstrap should be safe to run without loading heavy libs
ai_core.graph.bootstrap.bootstrap()

loaded = set(sys.modules.keys())
forbidden = ["torch", "transformers", "sentence_transformers", "fitz", "pdfplumber", "pikepdf"]

violations = []
for m in forbidden:
    if any(k == m or k.startswith(m + ".") for k in loaded):
        violations.append(m)

if violations:
    print(f"VIOLATIONS: {violations}")
    sys.exit(1)

print("SUCCESS: No forbidden modules loaded.")

"""

    # Run in a clean subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    # Remove DJANGO_SETTINGS_MODULE to prevent interference
    env.pop("DJANGO_SETTINGS_MODULE", None)

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    if result.returncode != 0:
        print("Import verification FAILED.")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        print("Import verification PASSED.")


if __name__ == "__main__":
    check_imports()
