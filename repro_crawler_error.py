import django

from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "customers",
            "documents",
            "cases",
        ],
        DATABASES={
            "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}
        },
        TENANT_MODEL="customers.Tenant",
        TENANT_DOMAIN_MODEL="customers.Domain",
    )
django.setup()
print(f"INSTALLED_APPS: {settings.INSTALLED_APPS}")

from ai_core.services import _dump_jsonable  # noqa: E402
from documents.parsers import ParsedResult, ParsedTextBlock  # noqa: E402
import json  # noqa: E402


def test_fix():
    print("Testing _dump_jsonable with ParsedResult...")

    block = ParsedTextBlock(text="Test", kind="paragraph")
    result = ParsedResult(text_blocks=(block,))

    print(f"Object type: {type(result)}")
    print(f"Is instance of ParsedResult? {isinstance(result, ParsedResult)}")

    try:
        safe = _dump_jsonable(result)
        print("Successfully converted to safe dict.")
        print(f"Safe dict keys: {safe.keys()}")

        json.dumps(safe)
        print("Successfully serialized to JSON.")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_fix()
