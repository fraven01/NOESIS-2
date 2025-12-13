"""Migrate crawler document metadata to correct structure.

Fixes:
1. Move provider from top-level to external_ref.provider
2. Set blob.media_type from content_type if missing
"""

from django.core.management.base import BaseCommand
from documents.models import Document


class Command(BaseCommand):
    help = "Migrate crawler document metadata to correct structure"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be changed without actually changing it",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]

        # Find all crawler documents
        docs = Document.objects.filter(source="crawler")
        total_docs = docs.count()

        self.stdout.write(f"Found {total_docs} crawler documents")

        fixed_external_ref = 0
        fixed_media_type = 0

        for doc in docs.iterator():
            metadata = doc.metadata or {}
            changed = False

            # Navigate to normalized_document structure
            normalized_doc = metadata.get("normalized_document", {})
            if not normalized_doc:
                continue

            meta = normalized_doc.get("meta", {})
            blob = normalized_doc.get("blob", {})

            # Fix #1: Move provider to external_ref
            # Check if provider is at top-level in metadata (old structure)
            if "provider" in metadata and "external_ref" not in meta:
                provider = metadata.pop("provider")
                origin_uri = meta.get("origin_uri", "")
                meta["external_ref"] = {
                    "provider": provider,
                    "external_id": f"{provider}::{origin_uri}",
                }
                normalized_doc["meta"] = meta
                changed = True
                fixed_external_ref += 1

                if dry_run:
                    self.stdout.write(
                        self.style.WARNING(
                            f"[DRY RUN] Would fix external_ref for {doc.id}"
                        )
                    )

            # Fix #2: Set media_type from content_type if missing
            if not blob.get("media_type"):
                # Try to get content_type from various locations
                content_type = (
                    metadata.get("content_type")
                    or meta.get("content_type")
                    or "application/octet-stream"
                )

                blob["media_type"] = content_type
                normalized_doc["blob"] = blob
                changed = True
                fixed_media_type += 1

                if dry_run:
                    self.stdout.write(
                        self.style.WARNING(
                            f"[DRY RUN] Would set media_type={content_type} for {doc.id}"
                        )
                    )

            # Save changes
            if changed:
                metadata["normalized_document"] = normalized_doc

                if not dry_run:
                    doc.metadata = metadata
                    doc.save(update_fields=["metadata"])

        # Summary
        if dry_run:
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n[DRY RUN] Would fix:\n"
                    f"  - {fixed_external_ref} external_refs\n"
                    f"  - {fixed_media_type} media_types"
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n✅ Successfully fixed:\n"
                    f"  - {fixed_external_ref} external_refs\n"
                    f"  - {fixed_media_type} media_types\n"
                    f"Total documents processed: {total_docs}"
                )
            )
