Param(
    [string[]]$Services = @('web', 'worker', 'agents-worker', 'ingestion-worker')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

$svcList = if ($Services -and $Services.Length -gt 0) { ($Services -join ' ') } else { 'web worker agents-worker ingestion-worker' }
Write-Host "[dev-restart] Restarting: $svcList"

try {
    iex "$compose restart $svcList"
}
catch {
    Write-Warning "[dev-restart] Restart failed (service may not exist yet). Trying up --no-deps --no-build."
    iex "$compose up -d --no-deps --no-build $svcList"
}

Write-Host '[dev-restart] Done'
