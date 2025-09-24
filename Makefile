.PHONY: jobs\:migrate jobs\:bootstrap tenant-new tenant-superuser jobs\:rag jobs\:rag\:health \
        seed-baseline seed-demo seed-heavy seed-chaos seed-wipe seed-check \
        load\:k6 load\:locust

PYTHON ?= python
MANAGE := $(PYTHON) manage.py
K6_BIN ?= k6
LOCUST_BIN ?= locust

jobs\:migrate:
	$(MANAGE) migrate_schemas --noinput

jobs\:bootstrap:
	@: $${DOMAIN:?Environment variable DOMAIN is required}
	$(MANAGE) bootstrap_public_tenant --domain "$$DOMAIN"

tenant-new:
	@: $${SCHEMA:?Environment variable SCHEMA is required}
	@: $${NAME:?Environment variable NAME is required}
	@: $${DOMAIN:?Environment variable DOMAIN is required}
	$(MANAGE) create_tenant --schema "$$SCHEMA" --name "$$NAME" --domain "$$DOMAIN"

tenant-superuser:
	@: $${SCHEMA:?Environment variable SCHEMA is required}
	@: $${USERNAME:?Environment variable USERNAME is required}
	@: $${PASSWORD:?Environment variable PASSWORD is required}
	@EMAIL_FLAG=""; \
	if [ -n "$$EMAIL" ]; then EMAIL_FLAG="--email \"$$EMAIL\""; fi; \
	$(MANAGE) create_tenant_superuser --schema "$$SCHEMA" --username "$$USERNAME" --password "$$PASSWORD" $$EMAIL_FLAG

jobs\:rag:
	@RAG_URL="$$RAG_DATABASE_URL"; \
	if [ -z "$$RAG_URL" ]; then RAG_URL="$$DATABASE_URL"; fi; \
	test -n "$$RAG_URL" || (echo "RAG_DATABASE_URL or DATABASE_URL must be set" >&2; exit 1); \
	psql "$$RAG_URL" -v ON_ERROR_STOP=1 -f docs/rag/schema.sql

jobs\:rag\:health:
	@RAG_URL="$$RAG_DATABASE_URL"; \
	if [ -z "$$RAG_URL" ]; then RAG_URL="$$DATABASE_URL"; fi; \
	test -n "$$RAG_URL" || (echo "RAG_DATABASE_URL or DATABASE_URL must be set" >&2; exit 1); \
	psql "$$RAG_URL" -v ON_ERROR_STOP=1 <<-'SQL'
	DO $$
	BEGIN
	    IF to_regnamespace('rag') IS NULL THEN
	        RAISE EXCEPTION 'Schema rag missing';
	    END IF;
	    IF to_regclass('rag.documents') IS NULL THEN
	        RAISE EXCEPTION 'Table rag.documents missing';
	    END IF;
	    IF to_regclass('rag.chunks') IS NULL THEN
	        RAISE EXCEPTION 'Table rag.chunks missing';
	    END IF;
	    IF to_regclass('rag.embeddings') IS NULL THEN
	        RAISE EXCEPTION 'Table rag.embeddings missing';
	    END IF;
	    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
	        RAISE EXCEPTION 'vector extension missing';
	    END IF;
	END;
	$$;
	SQL

seed-baseline: ; $(MANAGE) create_demo_data --profile baseline --seed 1337

seed-demo: ; $(MANAGE) create_demo_data --profile demo --seed 1337

seed-heavy: ; $(MANAGE) create_demo_data --profile heavy --seed 42

seed-chaos: ; $(MANAGE) create_demo_data --profile chaos --seed 99

seed-wipe: ; $(MANAGE) create_demo_data --wipe --include-org

seed-check: ; $(MANAGE) check_demo_data --profile demo --seed 1337

load\:k6:
	$(K6_BIN) run $(K6_ARGS) load/k6/script.js

load\:locust:
	$(LOCUST_BIN) -f load/locust/locustfile.py $(LOCUST_ARGS)
