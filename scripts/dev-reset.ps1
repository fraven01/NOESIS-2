Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

Write-Host '[dev-reset] Down + prune project volumes'
try { iex "$compose down -v --remove-orphans" } catch { Write-Warning $_ }

Write-Host '[dev-reset] Build fresh images'
iex "$compose build --no-cache --pull"

Write-Host '[dev-reset] Bring up base services'
iex "$compose up -d"

Write-Host '[dev-reset] Run jobs: migrate + bootstrap + rag'
iex 'npm run dev:init'

Write-Host '[dev-reset] Smoke checks'
iex 'npm run win:dev:check'

Write-Host '[dev-reset] Done'

