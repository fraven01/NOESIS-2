Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

Write-Host '[dev-reset-db] Down + prune database volume'
try { iex "$compose down -v db" } catch { Write-Warning $_ }

Write-Host '[dev-reset-db] Clearing local object store (.ai_core_store)'
if (Test-Path ".ai_core_store") {
    Remove-Item -Recurse -Force ".ai_core_store"
}

Write-Host '[dev-reset-db] Bring up base services'
iex "$compose up -d"

Write-Host '[dev-reset-db] Run jobs: migrate + bootstrap + rag'
iex 'npm run dev:init'

Write-Host '[dev-reset-db] Done'
