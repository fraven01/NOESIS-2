-- Prepare dedicated test database for isolated pytest runs (dev only)
-- Note: CREATE DATABASE cannot run inside a transaction/DO block.
-- Use psql's \gexec to conditionally create the DB idempotently.

SELECT 'CREATE DATABASE noesis2_test'
WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = 'noesis2_test'
)
\gexec
