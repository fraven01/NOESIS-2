#!/usr/bin/env python
"""Diagnose-Script für Collection-Duplicate-Problem im Dokumenten-Explorer.

Usage:
    python manage.py shell < scripts/diagnose_collection_issue.py

Oder direkt im Django Shell:
    exec(open('scripts/diagnose_collection_issue.py').read())
"""

from django.db.models import Count
from documents.models import Document, DocumentCollection, DocumentCollectionMembership
from customers.models import Tenant

print("=" * 80)
print("🔍 DIAGNOSE: Collection Duplicate Issue")
print("=" * 80)
print()

# 1. Tenant Info
print("1️⃣  TENANT INFO")
print("-" * 80)
try:
    public_tenant = Tenant.objects.get(schema_name="public")
    print(f"✅ Public Tenant gefunden: {public_tenant.name} (ID: {public_tenant.id})")
except Tenant.DoesNotExist:
    print("❌ Public Tenant nicht gefunden!")
    print("   Bitte Schema-Name anpassen.")
    public_tenant = None

print()

# 2. Collections Übersicht
print("2️⃣  COLLECTIONS OVERVIEW")
print("-" * 80)
if public_tenant:
    collections = DocumentCollection.objects.filter(tenant=public_tenant).order_by(
        "created_at"
    )
    print(f"Gefundene Collections: {collections.count()}")
    print()

    for i, coll in enumerate(collections, 1):
        print(f"   Collection {i}:")
        print(f"      - ID (PK):        {coll.id}")
        print(f"      - Collection ID:  {coll.collection_id}")
        print(f"      - Key:            '{coll.key}'")
        print(f"      - Name:           '{coll.name}'")
        print(f"      - Type:           {coll.type or '(none)'}")
        print(f"      - Created:        {coll.created_at}")

        # Zähle Dokumente in dieser Collection
        member_count = DocumentCollectionMembership.objects.filter(
            collection=coll
        ).count()
        print(f"      - Members:        {member_count} documents")
        print()
else:
    print("⚠️  Überspringe Collection-Analyse (kein Tenant)")
    print()

# 3. Duplikate finden
print("3️⃣  DUPLICATE DETECTION")
print("-" * 80)
if public_tenant:
    # Finde Collections mit gleicher collection_id
    duplicates = (
        DocumentCollection.objects.filter(tenant=public_tenant)
        .values("collection_id")
        .annotate(count=Count("id"))
        .filter(count__gt=1)
    )

    if duplicates:
        print(f"❌ GEFUNDEN: {len(duplicates)} Collection-IDs mit Duplikaten!")
        print()

        for dup in duplicates:
            coll_id = dup["collection_id"]
            count = dup["count"]
            print(f"   Collection ID: {coll_id} ({count} Einträge)")

            coll_set = DocumentCollection.objects.filter(
                tenant=public_tenant, collection_id=coll_id
            )
            for c in coll_set:
                member_count = DocumentCollectionMembership.objects.filter(
                    collection=c
                ).count()
                print(f"      - Key: '{c.key}' (PK: {c.id}, Members: {member_count})")
            print()
    else:
        print("✅ Keine Duplikate gefunden!")
        print()
else:
    print("⚠️  Überspringe Duplikat-Detektion (kein Tenant)")
    print()

# 4. Orphaned Documents
print("4️⃣  ORPHANED DOCUMENTS")
print("-" * 80)
if public_tenant:
    total_docs = Document.objects.filter(tenant=public_tenant).count()
    print(f"Gesamt Dokumente: {total_docs}")

    orphaned = Document.objects.filter(tenant=public_tenant).exclude(
        id__in=DocumentCollectionMembership.objects.values_list(
            "document_id", flat=True
        )
    )
    orphaned_count = orphaned.count()

    if orphaned_count > 0:
        print(f"❌ PROBLEM: {orphaned_count} Dokumente OHNE Collection-Membership!")
        print()
        print("   Sample (max 5):")
        for doc in orphaned[:5]:
            print(f"      - {doc.id} | {doc.source} | State: {doc.lifecycle_state}")
        print()
    else:
        print("✅ Alle Dokumente haben Memberships!")
        print()
else:
    print("⚠️  Überspringe Orphaned-Dokumente-Analyse (kein Tenant)")
    print()

# 5. UUID-Key Collections (Symptom des Bugs)
print("5️⃣  UUID-KEY COLLECTIONS (Bug-Symptom)")
print("-" * 80)
if public_tenant:
    import re

    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )

    uuid_key_collections = [
        c
        for c in DocumentCollection.objects.filter(tenant=public_tenant)
        if uuid_pattern.match(c.key)
    ]

    if uuid_key_collections:
        print(f"❌ GEFUNDEN: {len(uuid_key_collections)} Collections mit UUID als Key!")
        print("   (Das ist ein Symptom des Bugs!)")
        print()

        for c in uuid_key_collections:
            member_count = DocumentCollectionMembership.objects.filter(
                collection=c
            ).count()
            print(f"   - Key: '{c.key}'")
            print(f"     Name: '{c.name}'")
            print(f"     Members: {member_count} documents")
            print(f"     Collection ID: {c.collection_id}")
            print()
    else:
        print("✅ Keine Collections mit UUID als Key gefunden!")
        print()
else:
    print("⚠️  Überspringe UUID-Key-Analyse (kein Tenant)")
    print()

# 6. Membership-Analyse
print("6️⃣  MEMBERSHIP STATISTICS")
print("-" * 80)
if public_tenant:
    total_memberships = DocumentCollectionMembership.objects.filter(
        collection__tenant=public_tenant
    ).count()
    print(f"Gesamt Memberships: {total_memberships}")

    # Pro Collection
    print()
    print("Pro Collection:")
    for coll in DocumentCollection.objects.filter(tenant=public_tenant):
        count = DocumentCollectionMembership.objects.filter(collection=coll).count()
        if count > 0:
            print(f"   - '{coll.key}': {count} documents")
    print()
else:
    print("⚠️  Überspringe Membership-Analyse (kein Tenant)")
    print()

# 7. Zusammenfassung & Empfehlung
print("=" * 80)
print("📊 ZUSAMMENFASSUNG")
print("=" * 80)
print()

if public_tenant:
    has_duplicates = (
        DocumentCollection.objects.filter(tenant=public_tenant)
        .values("collection_id")
        .annotate(count=Count("id"))
        .filter(count__gt=1)
        .exists()
    )

    has_orphaned = (
        Document.objects.filter(tenant=public_tenant)
        .exclude(
            id__in=DocumentCollectionMembership.objects.values_list(
                "document_id", flat=True
            )
        )
        .exists()
    )

    has_uuid_keys = bool(
        [
            c
            for c in DocumentCollection.objects.filter(tenant=public_tenant)
            if re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", c.key
            )
        ]
    )

    issues = []
    if has_duplicates:
        issues.append("Duplikate Collections")
    if has_orphaned:
        issues.append("Orphaned Documents")
    if has_uuid_keys:
        issues.append("UUID-Key Collections")

    if issues:
        print("❌ PROBLEME GEFUNDEN:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("➡️  EMPFEHLUNG:")
        print("   1. Siehe DOCUMENT_EXPLORER_IST_ANALYSE.md für Details")
        print("   2. Implementiere Option 1 (Collection-Key statt UUID)")
        print("   3. Führe Cleanup-Script aus (in Analyse dokumentiert)")
        print()
    else:
        print("✅ Keine Probleme gefunden!")
        print("   Das System scheint korrekt zu funktionieren.")
        print()
else:
    print("⚠️  Konnte Diagnose nicht vollständig durchführen (kein Tenant)")
    print()

print("=" * 80)
print("Diagnose abgeschlossen!")
print("=" * 80)
