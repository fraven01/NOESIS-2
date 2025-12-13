import sys
import traceback
import os

# Set dummy env vars to avoid some config errors if checked at module level
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")

try:
    import ai_core.tests.test_crawler_ingestion_graph

    print("Import successful")
except Exception:
    traceback.print_exc()
