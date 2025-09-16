-- Initialize LiteLLM database for key management (dev only)
DO
$$
BEGIN
   PERFORM 1 FROM pg_database WHERE datname = 'litellm';
   IF NOT FOUND THEN
      PERFORM pg_sleep(0); -- no-op
      EXECUTE 'CREATE DATABASE litellm';
   END IF;
END
$$;

