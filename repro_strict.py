import os
import django
import time

# Wait a bit to let other logs flush
time.sleep(1)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()

from documents.contracts import InlineBlob  # noqa: E402
from documents.contract_utils import (  # noqa: E402
    normalize_media_type as utils_normalize,
)


def run_test():
    failures = []

    # Check 1: utils function
    try:
        raw = "text/html; charset=UTF-8"
        norm = utils_normalize(raw)
        if norm != "text/html":
            failures.append(f"utils_normalize returned '{norm}' instead of 'text/html'")
    except Exception as e:
        failures.append(f"utils_normalize crashed: {e}")

    # Check 2: Pydantic Validation
    try:
        _ = InlineBlob(
            type="inline",
            media_type="text/html; charset=UTF-8",
            base64="aGVsbG8=",
            sha256="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            size=5,
        )
    except Exception as e:
        failures.append(f"InlineBlob validation failed: {e}")

    print("-" * 40)
    if failures:
        print("RESULT: FAILURE")
        for f in failures:
            print(f"  - {f}")
    else:
        print("RESULT: SUCCESS")
    print("-" * 40)


if __name__ == "__main__":
    run_test()
