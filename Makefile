.PHONY: jobs\:migrate jobs\:bootstrap tenant-new tenant-superuser jobs\:rag jobs\:rag\:health \
        load\:k6 load\:locust

PYTHON ?= python
MANAGE := $(PYTHON) manage.py

OPENAPI_SCHEMA := docs/api/openapi.yaml

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
	DATABASE_URL="$$RAG_URL" $(MANAGE) sync_rag_schemas

jobs\:rag\:health:
	@RAG_URL="$$RAG_DATABASE_URL"; \
	if [ -z "$$RAG_URL" ]; then RAG_URL="$$DATABASE_URL"; fi; \
	test -n "$$RAG_URL" || (echo "RAG_DATABASE_URL or DATABASE_URL must be set" >&2; exit 1); \
	DATABASE_URL="$$RAG_URL" $(MANAGE) check_rag_schemas

.PHONY: schema sdk

schema:
	mkdir -p $(dir $(OPENAPI_SCHEMA))
	$(MANAGE) spectacular --format yaml --file $(OPENAPI_SCHEMA)

sdk: schema
	rm -rf clients/typescript
	npx --yes openapi-typescript-codegen@0.28.1 --input $(OPENAPI_SCHEMA) --output clients/typescript
	rm -rf clients/python
	openapi-python-client generate --path $(OPENAPI_SCHEMA) --output-path clients/python --overwrite

load\:k6:
	$(K6_BIN) run $(K6_ARGS) load/k6/script.js

load\:locust:
	$(LOCUST_BIN) -f load/locust/locustfile.py $(LOCUST_ARGS)

