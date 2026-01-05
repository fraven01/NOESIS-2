-- Index-Strategie wird später festgelegt.
-- Warum: Dieses Skript definiert das pgvector-Zielbild für RAG. Es ist idempotent und kann in der Pipeline-Stufe „Vector-Schema-Migrations“ ausgeführt werden.
-- Platzhalter: {{SCHEMA_NAME}} für das Ziel-Schema, {{VECTOR_DIM}} für die Embedding-Dimension.

BEGIN;

CREATE SCHEMA IF NOT EXISTS {{SCHEMA_NAME}};
SET search_path TO {{SCHEMA_NAME}}, public;

-- Optional: prepare dedicated roles and grants per schema/consumer
-- GRANT USAGE ON SCHEMA {{SCHEMA_NAME}} TO app_rw;
-- GRANT SELECT ON ALL TABLES IN SCHEMA {{SCHEMA_NAME}} TO app_ro;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA {{SCHEMA_NAME}}
--     GRANT SELECT ON TABLES TO app_ro;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA {{SCHEMA_NAME}}
--     GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_rw;
-- Duplicate-Guards wirken per Collection (NULL vs. NOT NULL).

-- Ensure extensions live in 'public' schema for global visibility
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;

DO $$
DECLARE
    v_ns oid;
    t_ns oid;
BEGIN
    SELECT extnamespace INTO v_ns FROM pg_extension WHERE extname = 'vector';
    IF v_ns IS NOT NULL AND v_ns <> 'public'::regnamespace THEN
        EXECUTE 'ALTER EXTENSION vector SET SCHEMA public';
    END IF;
    SELECT extnamespace INTO t_ns FROM pg_extension WHERE extname = 'pg_trgm';
    IF t_ns IS NOT NULL AND t_ns <> 'public'::regnamespace THEN
        EXECUTE 'ALTER EXTENSION pg_trgm SET SCHEMA public';
    END IF;
END $$;

-- Ensure pgvector is recent enough for HNSW + vector_cosine_ops
DO $$
DECLARE
    v_raw text;
    v text;
    maj int;
    min int;
    pat int;
    ver_num int;
BEGIN
    SELECT extversion INTO v_raw FROM pg_extension WHERE extname = 'vector';
    IF v_raw IS NULL THEN
        RAISE EXCEPTION 'pgvector extension is not installed';
    END IF;
    -- Extract leading semver (2- or 3-part) without using backreferences
    v := substring(v_raw from '^[0-9]+(?:\.[0-9]+){1,2}');
    maj := COALESCE(NULLIF(split_part(v, '.', 1), ''), '0')::int;
    min := COALESCE(NULLIF(split_part(v, '.', 2), ''), '0')::int;
    pat := COALESCE(NULLIF(split_part(v, '.', 3), ''), '0')::int;
    ver_num := maj * 10000 + min * 100 + pat;
    -- Require >= 0.5.0 (HNSW introduced; cosine operator classes supported)
    IF ver_num < 500 THEN
        RAISE EXCEPTION 'pgvector version % is too old. Require >= 0.5.0 for HNSW and vector_cosine_ops', v_raw;
    END IF;
END $$;

-- Collections kapseln optionale Dokument-Scopes je Tenant. Alle
-- Collection-IDs bleiben nullable in den Konsumententabellen, damit
-- Legacy-Bestände ohne Collection weiter funktionieren. Duplicate-
-- Guards berücksichtigen den Scope automatisch (NULL vs. NOT NULL).
CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.collections (
    tenant_id UUID NOT NULL,
    id UUID NOT NULL,
    slug TEXT,
    version_label TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, id)
);

CREATE UNIQUE INDEX IF NOT EXISTS collections_tenant_slug_uk
    ON {{SCHEMA_NAME}}.collections (tenant_id, slug)
    WHERE slug IS NOT NULL AND slug <> '';

CREATE UNIQUE INDEX IF NOT EXISTS collections_tenant_version_label_uk
    ON {{SCHEMA_NAME}}.collections (tenant_id, version_label)
    WHERE version_label IS NOT NULL AND version_label <> '';

-- Dokumente, Chunks und Embeddings speichern weiterhin `collection_id`
-- als optionale Spalte für schnelle Filter. Die inhaltliche Identität
-- eines Dokuments wird jedoch pro Tenant/Source dedupliziert; die
-- Zuordnung zu Collections erfolgt über die Relation
-- `document_collections`.
CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.documents (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    collection_id UUID,
    workflow_id TEXT,
    source TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    lifecycle TEXT NOT NULL DEFAULT 'active',
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.document_collections (
    document_id UUID NOT NULL REFERENCES {{SCHEMA_NAME}}.documents(id) ON DELETE CASCADE,
    collection_id UUID NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    added_by TEXT NOT NULL DEFAULT 'system',
    PRIMARY KEY (document_id, collection_id)
);

CREATE INDEX IF NOT EXISTS document_collections_collection_idx
    ON {{SCHEMA_NAME}}.document_collections (collection_id);

CREATE INDEX IF NOT EXISTS document_collections_document_idx
    ON {{SCHEMA_NAME}}.document_collections (document_id);

CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON {{SCHEMA_NAME}}.documents USING GIN (metadata);

ALTER TABLE {{SCHEMA_NAME}}.documents ADD COLUMN IF NOT EXISTS external_id TEXT;

ALTER TABLE {{SCHEMA_NAME}}.documents ADD COLUMN IF NOT EXISTS collection_id UUID;

ALTER TABLE {{SCHEMA_NAME}}.documents ADD COLUMN IF NOT EXISTS workflow_id TEXT;

ALTER TABLE {{SCHEMA_NAME}}.documents
    ADD COLUMN IF NOT EXISTS lifecycle TEXT NOT NULL DEFAULT 'active';

ALTER TABLE {{SCHEMA_NAME}}.documents
    ALTER COLUMN collection_id DROP NOT NULL;

CREATE INDEX IF NOT EXISTS documents_hash_idx
    ON {{SCHEMA_NAME}}.documents (hash);

DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_collection_idx;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_collection_workflow_idx;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_hash_idx;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_hash_null_collection_idx;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_workflow_hash_null_collection_idx;

CREATE INDEX IF NOT EXISTS documents_tenant_workflow_idx
    ON {{SCHEMA_NAME}}.documents (tenant_id, workflow_id);

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_source_hash_idx
    ON {{SCHEMA_NAME}}.documents (tenant_id, source, hash)
    WHERE workflow_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_workflow_source_hash_idx
    ON {{SCHEMA_NAME}}.documents (tenant_id, workflow_id, source, hash)
    WHERE workflow_id IS NOT NULL;

DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_external_id_uk;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_external_id_null_collection_uk;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_collection_external_id_uk;
DROP INDEX IF EXISTS {{SCHEMA_NAME}}.documents_tenant_collection_workflow_external_id_uk;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_external_id_uk
    ON {{SCHEMA_NAME}}.documents (tenant_id, external_id)
    WHERE workflow_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_workflow_external_id_uk
    ON {{SCHEMA_NAME}}.documents (tenant_id, workflow_id, external_id)
    WHERE workflow_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES {{SCHEMA_NAME}}.documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    tenant_id UUID,
    collection_id UUID
);

ALTER TABLE {{SCHEMA_NAME}}.chunks
    ADD COLUMN IF NOT EXISTS text_norm TEXT
        GENERATED ALWAYS AS (lower(regexp_replace(text, '\s+', ' ', 'g'))) STORED;

ALTER TABLE {{SCHEMA_NAME}}.chunks
    ADD COLUMN IF NOT EXISTS text_tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector('simple', lower(regexp_replace(text, '\s+', ' ', 'g')))
        ) STORED;

ALTER TABLE {{SCHEMA_NAME}}.chunks ADD COLUMN IF NOT EXISTS tenant_id UUID;

ALTER TABLE {{SCHEMA_NAME}}.chunks ADD COLUMN IF NOT EXISTS collection_id UUID;

CREATE INDEX IF NOT EXISTS chunks_document_ord_idx
    ON {{SCHEMA_NAME}}.chunks (document_id, ord);

CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
    ON {{SCHEMA_NAME}}.chunks USING GIN ((metadata) jsonb_path_ops);

-- Targeted index to accelerate equality filters on common metadata keys
CREATE INDEX IF NOT EXISTS chunks_metadata_case_idx
    ON {{SCHEMA_NAME}}.chunks ((metadata->>'case'));

CREATE INDEX IF NOT EXISTS chunks_text_norm_trgm_idx
    ON {{SCHEMA_NAME}}.chunks USING GIN (text_norm gin_trgm_ops);

CREATE INDEX IF NOT EXISTS chunks_text_tsv_idx
    ON {{SCHEMA_NAME}}.chunks USING GIN (text_tsv);

CREATE INDEX IF NOT EXISTS chunks_document_collection_idx
    ON {{SCHEMA_NAME}}.chunks (document_id, collection_id);

CREATE INDEX IF NOT EXISTS chunks_tenant_collection_idx
    ON {{SCHEMA_NAME}}.chunks (tenant_id, collection_id);

CREATE INDEX IF NOT EXISTS chunks_collection_idx
    ON {{SCHEMA_NAME}}.chunks (collection_id)
    WHERE collection_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS chunks_embedding_profile_scope_idx
    ON {{SCHEMA_NAME}}.chunks (
        (metadata->>'embedding_profile'),
        (metadata->>'case'),
        collection_id
    )
    WHERE metadata ? 'embedding_profile';

CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.embeddings (
    id UUID PRIMARY KEY,
    chunk_id UUID NOT NULL REFERENCES {{SCHEMA_NAME}}.chunks(id) ON DELETE CASCADE,
    embedding vector({{VECTOR_DIM}}) NOT NULL,
    tenant_id UUID,
    collection_id UUID
);

ALTER TABLE {{SCHEMA_NAME}}.embeddings ADD COLUMN IF NOT EXISTS tenant_id UUID;

ALTER TABLE {{SCHEMA_NAME}}.embeddings ADD COLUMN IF NOT EXISTS collection_id UUID;

CREATE UNIQUE INDEX IF NOT EXISTS embeddings_chunk_idx
    ON {{SCHEMA_NAME}}.embeddings (chunk_id);

CREATE INDEX IF NOT EXISTS embeddings_collection_idx
    ON {{SCHEMA_NAME}}.embeddings (collection_id)
    WHERE collection_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS embeddings_tenant_collection_idx
    ON {{SCHEMA_NAME}}.embeddings (tenant_id, collection_id);

CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.embedding_cache (
    text_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    embedding vector({{VECTOR_DIM}}) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() + INTERVAL '90 days'),
    PRIMARY KEY (text_hash, model_version)
);

CREATE INDEX IF NOT EXISTS embedding_cache_expires_idx
    ON {{SCHEMA_NAME}}.embedding_cache (expires_at);

DO $$
BEGIN
    ALTER TABLE {{SCHEMA_NAME}}.documents
        ADD CONSTRAINT documents_collection_fk
        FOREIGN KEY (tenant_id, collection_id)
        REFERENCES {{SCHEMA_NAME}}.collections (tenant_id, id)
        DEFERRABLE INITIALLY IMMEDIATE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE {{SCHEMA_NAME}}.chunks
        ADD CONSTRAINT chunks_collection_fk
        FOREIGN KEY (tenant_id, collection_id)
        REFERENCES {{SCHEMA_NAME}}.collections (tenant_id, id)
        DEFERRABLE INITIALLY IMMEDIATE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE {{SCHEMA_NAME}}.embeddings
        ADD CONSTRAINT embeddings_collection_fk
        FOREIGN KEY (tenant_id, collection_id)
        REFERENCES {{SCHEMA_NAME}}.collections (tenant_id, id)
        DEFERRABLE INITIALLY IMMEDIATE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

DROP INDEX IF EXISTS {{SCHEMA_NAME}}.embeddings_embedding_hnsw;
CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw
    ON {{SCHEMA_NAME}}.embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 200);

-- Optional index stubs (enable as needed per environment)
-- CREATE INDEX IF NOT EXISTS embeddings_embedding_ivfflat
--     ON {{SCHEMA_NAME}}.embeddings USING ivfflat (embedding vector_l2_ops)
--     WITH (lists = 100);
-- CREATE INDEX IF NOT EXISTS chunks_text_ivfflat
--     ON {{SCHEMA_NAME}}.chunks USING ivfflat ((text_norm::vector) vector_l2_ops)
--     WITH (lists = 100);

COMMIT;

ANALYZE {{SCHEMA_NAME}}.documents;
ANALYZE {{SCHEMA_NAME}}.chunks;
ANALYZE {{SCHEMA_NAME}}.embeddings;

-- Hinweise für Migrationen:
-- * Führe Index-Updates mit `CREATE INDEX CONCURRENTLY` in Prod aus, falls Downtime vermieden werden muss.
-- * Nach großen Löschläufen: `VACUUM (VERBOSE, ANALYZE) {{SCHEMA_NAME}}.embeddings;`
-- * Bei Schemaänderungen stets `IF NOT EXISTS`/`ADD COLUMN IF NOT EXISTS` nutzen, damit Wiederholungen idempotent bleiben.
