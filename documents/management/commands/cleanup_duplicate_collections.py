"""Cleanup duplicate collections created by the collection_id lookup bug.

This management command finds and merges duplicate collections that were
created due to the bug where _resolve_collection_reference created new
collections with UUID as key instead of finding existing ones.

The bug:
1. Collection created with key="manual-search", collection_id=<uuid>
2. Ingestion passes collection_id=<uuid>
3. _resolve_collection_reference doesn't find it
4. New collection created with key=<uuid>, collection_id=<uuid>
5. Documents linked to wrong collection

This script:
1. Finds duplicate collections (same collection_id, different keys)
2. Identifies the "correct" collection (semantic key, not UUID)
3. Moves DocumentCollectionMembership to correct collection
4. Deletes duplicate collections

Usage:
    python manage.py cleanup_duplicate_collections
    python manage.py cleanup_duplicate_collections --tenant=public
    python manage.py cleanup_duplicate_collections --dry-run
"""

import re
from django.core.management.base import BaseCommand
from django.db import transaction
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.models import DocumentCollection, DocumentCollectionMembership
from common.logging import get_logger

logger = get_logger(__name__)

# Regex to match UUID strings (potential duplicate keys)
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class Command(BaseCommand):
    """Cleanup duplicate collections from collection_id lookup bug."""

    help = "Find and merge duplicate collections"

    def add_arguments(self, parser):
        parser.add_argument(
            "--tenant",
            type=str,
            help="Specific tenant schema name (default: all tenants)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be cleaned up without actually doing it",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show detailed information about each duplicate",
        )

    def handle(self, *args, **options):
        tenant_filter = options.get("tenant")
        dry_run = options.get("dry_run", False)
        verbose = options.get("verbose", False)

        if tenant_filter:
            tenants = Tenant.objects.filter(schema_name=tenant_filter)
            if not tenants.exists():
                self.stdout.write(
                    self.style.ERROR(f"Tenant '{tenant_filter}' not found")
                )
                return
        else:
            tenants = Tenant.objects.all()

        total_tenants = tenants.count()
        total_duplicates = 0
        total_merged = 0
        total_memberships_moved = 0

        self.stdout.write(
            self.style.SUCCESS(
                f"Scanning {total_tenants} tenant(s) for duplicate collections..."
            )
        )

        for tenant in tenants:
            with schema_context(tenant.schema_name):
                self.stdout.write(f"\nProcessing tenant: {tenant.schema_name}")

                # Find collections with same collection_id
                duplicates = self._find_duplicates(tenant)

                if not duplicates:
                    self.stdout.write(self.style.SUCCESS("  ✅ No duplicates found"))
                    continue

                total_duplicates += len(duplicates)
                self.stdout.write(
                    self.style.WARNING(
                        f"  ⚠️  Found {len(duplicates)} duplicate collection_id(s)"
                    )
                )

                for collection_id, collections in duplicates.items():
                    self.stdout.write(f"\n  Collection ID: {collection_id}")

                    # Identify correct and duplicate collections
                    correct, duplicates_list = self._classify_collections(collections)

                    if not correct:
                        self.stdout.write(
                            self.style.ERROR(
                                "    ❌ Could not identify correct collection (skipping)"
                            )
                        )
                        continue

                    self.stdout.write(
                        f"    Correct collection: key='{correct.key}' (PK: {correct.id})"
                    )

                    for dup in duplicates_list:
                        member_count = DocumentCollectionMembership.objects.filter(
                            collection=dup
                        ).count()

                        self.stdout.write(
                            f"    Duplicate: key='{dup.key}' (PK: {dup.id}, {member_count} members)"
                        )

                        if verbose:
                            self.stdout.write(f"      - Name: '{dup.name}'")
                            self.stdout.write(f"      - Created: {dup.created_at}")
                            self.stdout.write(f"      - Type: {dup.type or '(none)'}")

                        if dry_run:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"      [DRY RUN] Would move {member_count} memberships and delete"
                                )
                            )
                            total_memberships_moved += member_count
                        else:
                            # Move memberships and delete duplicate
                            moved = self._merge_collections(correct, dup)
                            total_memberships_moved += moved
                            total_merged += 1

                            self.stdout.write(
                                self.style.SUCCESS(
                                    f"      ✅ Moved {moved} memberships and deleted duplicate"
                                )
                            )

        # Summary
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("Cleanup Summary:"))
        self.stdout.write(f"  Total tenants scanned: {total_tenants}")
        self.stdout.write(f"  Duplicate collection_ids found: {total_duplicates}")
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"  Would merge: {len([d for dup_list in [self._classify_collections(c)[1] for c in [v for _, v in self._find_all_duplicates(tenants)]] for d in dup_list])}"
                )
            )
            self.stdout.write(
                self.style.WARNING(
                    f"  Would move memberships: {total_memberships_moved}"
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f"  Collections merged: {total_merged}")
            )
            self.stdout.write(
                self.style.SUCCESS(f"  Memberships moved: {total_memberships_moved}")
            )
        self.stdout.write("=" * 80)

        if dry_run:
            self.stdout.write(
                self.style.WARNING("\nℹ️  This was a DRY RUN - no changes were made")
            )
            self.stdout.write("Run without --dry-run to actually clean up duplicates")

    def _find_duplicates(self, tenant):
        """Find collections with duplicate collection_ids."""
        from django.db.models import Count

        # Find collection_ids that appear more than once
        duplicate_ids = (
            DocumentCollection.objects.filter(tenant=tenant)
            .values("collection_id")
            .annotate(count=Count("id"))
            .filter(count__gt=1)
        )

        duplicates = {}
        for item in duplicate_ids:
            collection_id = item["collection_id"]
            collections = list(
                DocumentCollection.objects.filter(
                    tenant=tenant, collection_id=collection_id
                )
            )
            duplicates[collection_id] = collections

        return duplicates

    def _find_all_duplicates(self, tenants):
        """Helper to find all duplicates across tenants (for summary)."""
        all_duplicates = []
        for tenant in tenants:
            with schema_context(tenant.schema_name):
                duplicates = self._find_duplicates(tenant)
                all_duplicates.extend(duplicates.items())
        return all_duplicates

    def _classify_collections(self, collections):
        """Identify correct collection vs duplicates.

        Returns: (correct_collection, [duplicate_collections])

        Heuristics:
        1. Prefer collections with semantic keys (not UUIDs)
        2. Among semantic keys, prefer "manual-search" or similar
        3. If all are UUIDs, prefer oldest
        """
        if len(collections) == 1:
            return collections[0], []

        # Separate UUID-key collections from semantic-key collections
        uuid_key_collections = []
        semantic_key_collections = []

        for coll in collections:
            if UUID_PATTERN.match(coll.key):
                uuid_key_collections.append(coll)
            else:
                semantic_key_collections.append(coll)

        # Prefer semantic keys
        if semantic_key_collections:
            # If multiple semantic keys, prefer system collections
            system_collections = [
                c
                for c in semantic_key_collections
                if c.key in ("manual-search", "manual-uploads", "web-crawler")
            ]

            if system_collections:
                # Prefer oldest system collection
                correct = min(system_collections, key=lambda c: c.created_at)
            else:
                # Prefer oldest semantic key
                correct = min(semantic_key_collections, key=lambda c: c.created_at)

            # All others are duplicates
            duplicates = [c for c in collections if c.id != correct.id]
            return correct, duplicates

        # If all are UUID keys, prefer oldest
        correct = min(uuid_key_collections, key=lambda c: c.created_at)
        duplicates = [c for c in uuid_key_collections if c.id != correct.id]
        return correct, duplicates

    @transaction.atomic
    def _merge_collections(self, correct, duplicate):
        """Move all memberships from duplicate to correct, then delete duplicate.

        Returns: Number of memberships moved
        """
        # Get all memberships for the duplicate
        memberships = DocumentCollectionMembership.objects.filter(collection=duplicate)

        moved_count = 0
        for membership in memberships:
            # Check if membership already exists for this document in correct collection
            existing = DocumentCollectionMembership.objects.filter(
                document=membership.document, collection=correct
            ).first()

            if existing:
                # Membership already exists, just delete duplicate
                membership.delete()
            else:
                # Move membership to correct collection
                membership.collection = correct
                membership.save()
                moved_count += 1

        # Delete the duplicate collection
        duplicate.delete()

        logger.info(
            "cleanup_duplicate_collections.merged",
            extra={
                "correct_collection_id": str(correct.id),
                "correct_key": correct.key,
                "duplicate_collection_id": str(duplicate.id),
                "duplicate_key": duplicate.key,
                "memberships_moved": moved_count,
            },
        )

        return moved_count
