Param(
    [switch]$IncludeInfrastructure,
    [string[]]$Services = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

$defaultServices = @('web', 'worker', 'beat')
$infrastructureServices = @('db', 'redis', 'litellm')

if ($Services -and $Services.Length -gt 0) {
    $serviceList = $Services
}
else {
    $serviceList = $defaultServices
}

if ($IncludeInfrastructure.IsPresent) {
    $serviceList += $infrastructureServices
}

$serviceList = $serviceList | Select-Object -Unique
$svcList = if ($serviceList.Count -gt 0) { ($serviceList -join ' ') } else { ($defaultServices -join ' ') }
Write-Host "[dev-restart] Restarting: $svcList"

try {
    iex "$compose restart $svcList"
}
catch {
    Write-Warning "[dev-restart] Restart failed (service may not exist yet). Trying up --no-deps --no-build."
    iex "$compose up -d --no-deps --no-build $svcList"
}

Write-Host '[dev-restart] Done'
