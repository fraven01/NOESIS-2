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

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    source TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_hash_idx
    ON documents (tenant_id, hash);

CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents USING GIN (metadata);

ALTER TABLE {{SCHEMA_NAME}}.documents ADD COLUMN IF NOT EXISTS external_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_external_id_uk
    ON {{SCHEMA_NAME}}.documents (tenant_id, external_id);

CREATE INDEX IF NOT EXISTS documents_hash_idx
    ON {{SCHEMA_NAME}}.documents (hash);

CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);

ALTER TABLE {{SCHEMA_NAME}}.chunks
    ADD COLUMN IF NOT EXISTS text_norm TEXT
        GENERATED ALWAYS AS (lower(regexp_replace(text, '\s+', ' ', 'g'))) STORED;

CREATE INDEX IF NOT EXISTS chunks_document_ord_idx
    ON {{SCHEMA_NAME}}.chunks (document_id, ord);

CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
    ON {{SCHEMA_NAME}}.chunks USING GIN ((metadata) jsonb_path_ops);

-- Targeted index to accelerate equality filters on common metadata keys
CREATE INDEX IF NOT EXISTS chunks_metadata_case_idx
    ON {{SCHEMA_NAME}}.chunks ((metadata->>'case'));

CREATE INDEX IF NOT EXISTS chunks_text_norm_trgm_idx
    ON {{SCHEMA_NAME}}.chunks USING GIN (text_norm gin_trgm_ops);

CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.embeddings (
    id UUID PRIMARY KEY,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector({{VECTOR_DIM}}) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS embeddings_chunk_idx
    ON {{SCHEMA_NAME}}.embeddings (chunk_id);

DROP INDEX IF EXISTS embeddings_embedding_hnsw;
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
