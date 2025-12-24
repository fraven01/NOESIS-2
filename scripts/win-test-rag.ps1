$ErrorActionPreference = "Stop"

$env:AI_CORE_TEST_DATABASE_URL = "postgresql://noesis2:noesis2@db:5432/noesis2_test"
$env:RAG_DATABASE_URL = "postgresql://noesis2:noesis2@db:5432/noesis2_test"

docker compose -f docker-compose.dev.yml run --rm web python -m pytest -q
