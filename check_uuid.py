import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")

import django


def main() -> None:
    django.setup()
    from ai_core.rag.collections import manual_collection_uuid

    schema = "autotest"
    expected = str(manual_collection_uuid(schema))
    actual = "49f6a1b0-99b9-5160-b9cc-92cc7ef9c319"

    print(f"Schema: {schema}")
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"Match:    {expected == actual}")


if __name__ == "__main__":
    main()
