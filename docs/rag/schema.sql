-- Index-Strategie wird später festgelegt.
-- Warum: Dieses Skript definiert das pgvector-Zielbild für RAG. Es ist idempotent und kann in der Pipeline-Stufe „Vector-Schema-Migrations“ ausgeführt werden.

BEGIN;

CREATE SCHEMA IF NOT EXISTS rag;
SET search_path TO rag, public;

CREATE EXTENSION IF NOT EXISTS vector;

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

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS chunks_document_ord_idx
    ON chunks (document_id, ord);

CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
    ON chunks USING GIN (metadata);

CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS embeddings_chunk_idx
    ON embeddings (chunk_id);

CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw
    ON embeddings USING hnsw (embedding vector_l2_ops);

COMMIT;

ANALYZE rag.documents;
ANALYZE rag.chunks;
ANALYZE rag.embeddings;

-- Hinweise für Migrationen:
-- * Führe Index-Updates mit `CREATE INDEX CONCURRENTLY` in Prod aus, falls Downtime vermieden werden muss.
-- * Nach großen Löschläufen: `VACUUM (VERBOSE, ANALYZE) rag.embeddings;`
-- * Bei Schemaänderungen stets `IF NOT EXISTS`/`ADD COLUMN IF NOT EXISTS` nutzen, damit Wiederholungen idempotent bleiben.

-- TODO (später):
-- * Auswahl IVFFLAT vs. HNSW abhängig von Datenvolumen und Latenz.
-- * Entscheidung folgt nach Messung.
