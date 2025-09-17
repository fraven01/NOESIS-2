-- Initialize LiteLLM database for key management (dev only)
-- Note: CREATE DATABASE cannot run inside a transaction/DO block.
-- Use psql's \gexec to conditionally create the DB idempotently.

SELECT 'CREATE DATABASE litellm'
WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = 'litellm'
)
\gexec
