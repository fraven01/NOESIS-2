Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

Write-Host '[dev-restart] Restarting web + worker'
iex "$compose restart web worker"

Write-Host '[dev-restart] Done'
