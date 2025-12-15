#!/usr/bin/env python
"""Debug script to check document blob and parser compatibility."""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")

import django
django.setup()

from uuid import UUID
from django_tenants.utils import schema_context
from customers.models import Tenant
from documents.models import Document

DOC_ID = "fd168f2b-3289-47a8-8b76-0e8271374a66"
TENANT_SCHEMA = "dev"

def main():
    doc_uuid = UUID(DOC_ID)
    print(f"=== Checking document blob for: {DOC_ID} ===\n")
    
    tenant = Tenant.objects.get(schema_name=TENANT_SCHEMA)
    
    with schema_context(TENANT_SCHEMA):
        doc = Document.objects.get(id=doc_uuid)
        
        print(f"Document found: {doc.id}")
        print(f"Source: {doc.source}")
        print(f"Hash: {doc.hash}")
        print(f"Lifecycle: {doc.lifecycle_state}")
        
        print("\n=== Metadata ===")
        metadata = doc.metadata
        
        if metadata:
            normalized_doc = metadata.get("normalized_document", {})
            
            blob = normalized_doc.get("blob", {})
            print(f"Blob type: {blob.get('type')}")
            print(f"Blob media_type: {blob.get('media_type')}")
            print(f"Blob size: {blob.get('size')}")
            print(f"Blob sha256: {blob.get('sha256', '')[:20]}...")
            print(f"Blob uri: {blob.get('uri')}")
            print(f"Blob base64 present: {'base64' in blob and len(blob.get('base64', '')) > 0}")
            
            ref = normalized_doc.get("ref", {})
            print(f"\nRef tenant_id: {ref.get('tenant_id')}")
            print(f"Ref workflow_id: {ref.get('workflow_id')}")
            print(f"Ref document_id: {ref.get('document_id')}")
            print(f"Ref collection_id: {ref.get('collection_id')}")
            
            meta = normalized_doc.get("meta", {})
            print(f"\nMeta title: {meta.get('title')}")
            print(f"Meta external_ref: {meta.get('external_ref')}")
            
    # Check parser registry
    print("\n=== Parser Registry ===")
    from documents import (
        MarkdownDocumentParser,
        HtmlDocumentParser,
        DocxDocumentParser,
        PptxDocumentParser,
        PdfDocumentParser,
        ImageDocumentParser,
        ParserRegistry,
    )
    
    registry = ParserRegistry()
    registry.register(MarkdownDocumentParser())
    registry.register(HtmlDocumentParser())
    registry.register(DocxDocumentParser())
    registry.register(PptxDocumentParser())
    registry.register(PdfDocumentParser())
    registry.register(ImageDocumentParser())
    
    print("Registered parsers:")
    for parser in registry._parsers:
        mime_types = getattr(parser, 'supported_mime_types', None) or getattr(parser, 'MIME_TYPES', None)
        print(f"  - {type(parser).__name__}: {mime_types}")
    
    # Check if media_type matches
    if metadata:
        blob = metadata.get("normalized_document", {}).get("blob", {})
        media_type = blob.get("media_type", "")
        print(f"\nDocument media_type: {media_type}")
        
        matched = False
        for parser in registry._parsers:
            mime_types = getattr(parser, 'supported_mime_types', None) or getattr(parser, 'MIME_TYPES', [])
            if media_type in mime_types:
                print(f"MATCH: {type(parser).__name__} supports {media_type}")
                matched = True
                break
        
        if not matched:
            print(f"NO MATCH: No parser supports {media_type}")
            print("This is the cause of 'no_parser_found' error!")

if __name__ == "__main__":
    main()
