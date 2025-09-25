Param(
    [switch]$WithFrontend
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

Write-Host '[dev-rebuild] Building web and worker images'
iex "$compose build web worker"

if ($WithFrontend) {
    Write-Host '[dev-rebuild] Building frontend image'
    iex "$compose build frontend"
}
