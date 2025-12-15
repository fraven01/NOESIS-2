#!/usr/bin/env python
"""Debug script to check pipeline state and ingestion status for a document."""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")

import django

django.setup()

import json
from uuid import UUID
from pathlib import Path
from ai_core.infra import object_store
from ai_core.ingestion import _status_store_path, _meta_store_path, _load_pipeline_state

DOC_ID = "fd168f2b-3289-47a8-8b76-0e8271374a66"
TENANT = "dev"
CASE = "upload"  # Default case for uploads


def main():
    print(f"=== Checking pipeline state for document: {DOC_ID} ===\n")
    print(f"Tenant: {TENANT}, Case: {CASE}\n")

    # Check status file
    status_path = _status_store_path(TENANT, CASE, DOC_ID)
    print(f"Status file path: {status_path}")
    full_status_path = object_store.BASE_PATH / status_path
    print(f"Full path: {full_status_path}")
    print(f"Exists: {full_status_path.exists()}")

    if full_status_path.exists():
        try:
            with open(full_status_path) as f:
                status_content = json.load(f)
            print(
                f"Status content:\n{json.dumps(status_content, indent=2, default=str)}"
            )
        except Exception as e:
            print(f"Error reading status: {e}")
    else:
        print(
            "No status file found - task was either never started or uses different path"
        )

    print()

    # Check meta file
    meta_path = _meta_store_path(TENANT, CASE, DOC_ID)
    print(f"Meta file path: {meta_path}")
    full_meta_path = object_store.BASE_PATH / meta_path
    print(f"Full path: {full_meta_path}")
    print(f"Exists: {full_meta_path.exists()}")

    if full_meta_path.exists():
        try:
            with open(full_meta_path) as f:
                meta_content = json.load(f)
            print(f"Meta content:\n{json.dumps(meta_content, indent=2, default=str)}")
        except Exception as e:
            print(f"Error reading meta: {e}")

    print()

    # List all files in uploads dir
    uploads_dir = object_store.BASE_PATH / TENANT / CASE / "uploads"
    print(f"Uploads directory: {uploads_dir}")
    print(f"Exists: {uploads_dir.exists()}")

    if uploads_dir.exists():
        files = list(uploads_dir.glob(f"{DOC_ID}*"))
        print(f"Files matching doc ID: {len(files)}")
        for f in files[:10]:
            print(f"  - {f.name}")

    # Check using ingestion load function
    print("\n=== Using _load_pipeline_state ===")
    state = _load_pipeline_state(TENANT, CASE, DOC_ID)
    print(f"Loaded state:\n{json.dumps(state, indent=2, default=str)}")


if __name__ == "__main__":
    main()
