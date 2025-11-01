-- Migration: Introduce document_collections relation and update dedupe indexes
-- Usage: run with psql -v SCHEMA_NAME='<schema>' -f this_file.sql
BEGIN;

SET search_path TO {{SCHEMA_NAME}}, public;

-- Create relation table for document to collection associations
CREATE TABLE IF NOT EXISTS document_collections (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id UUID NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    added_by TEXT NOT NULL DEFAULT 'system',
    PRIMARY KEY (document_id, collection_id)
);

CREATE INDEX IF NOT EXISTS document_collections_collection_idx
    ON document_collections (collection_id);

CREATE INDEX IF NOT EXISTS document_collections_document_idx
    ON document_collections (document_id);

-- Refresh dedupe indexes to use tenant/source identity
DROP INDEX IF EXISTS documents_tenant_collection_idx;
DROP INDEX IF EXISTS documents_tenant_collection_workflow_idx;
DROP INDEX IF EXISTS documents_tenant_hash_idx;
DROP INDEX IF EXISTS documents_tenant_hash_null_collection_idx;
DROP INDEX IF EXISTS documents_tenant_workflow_hash_null_collection_idx;

CREATE INDEX IF NOT EXISTS documents_tenant_workflow_idx
    ON documents (tenant_id, workflow_id);

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_source_hash_idx
    ON documents (tenant_id, source, hash)
    WHERE workflow_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_workflow_source_hash_idx
    ON documents (tenant_id, workflow_id, source, hash)
    WHERE workflow_id IS NOT NULL;

DROP INDEX IF EXISTS documents_tenant_external_id_uk;
DROP INDEX IF EXISTS documents_tenant_external_id_null_collection_uk;
DROP INDEX IF EXISTS documents_tenant_collection_external_id_uk;
DROP INDEX IF EXISTS documents_tenant_collection_workflow_external_id_uk;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_external_id_uk
    ON documents (tenant_id, external_id)
    WHERE workflow_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_workflow_external_id_uk
    ON documents (tenant_id, workflow_id, external_id)
    WHERE workflow_id IS NOT NULL;

COMMIT;
