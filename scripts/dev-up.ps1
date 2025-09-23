Param(
    [string]$DevTenantSchema = $env:DEV_TENANT_SCHEMA,
    [string]$DevTenantName = $env:DEV_TENANT_NAME,
    [string]$DevDomain = $env:DEV_DOMAIN,
    [string]$DevSuperuserUsername = $env:DEV_SUPERUSER_USERNAME,
    [string]$DevSuperuserEmail = $env:DEV_SUPERUSER_EMAIL,
    [string]$DevSuperuserPassword = $env:DEV_SUPERUSER_PASSWORD
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path '.env' -PathType Leaf)) {
    Write-Error '[dev-up] Keine .env im Projektstamm gefunden. Bitte .env.example nach .env kopieren und Werte anpassen.'
}

if (-not $DevTenantSchema) { $DevTenantSchema = 'dev' }
if (-not $DevTenantName) { $DevTenantName = 'Dev Tenant' }
if (-not $DevDomain) { $DevDomain = 'dev.localhost' }
if (-not $DevSuperuserUsername) { $DevSuperuserUsername = 'admin' }
if (-not $DevSuperuserEmail) { $DevSuperuserEmail = 'admin@example.com' }
if (-not $DevSuperuserPassword) { $DevSuperuserPassword = 'admin123' }

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

Write-Host "[dev-up] Bringing up services (db, redis, litellm, web, worker)"
Invoke-Expression "$compose up -d"

Write-Host "[dev-up] Waiting for web to respond (warm-up)"
$ok = $false
for ($i=0; $i -lt 20; $i++) {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/ai/ping/' -Method GET
    if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { Write-Host "[dev-up] Web responded with HTTP $($resp.StatusCode)"; $ok=$true; break }
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { Write-Host "[dev-up] Web not responding yet, continuing" }

Write-Host "[dev-up] Init jobs: migrate + bootstrap"
Invoke-Expression 'npm run dev:init'

Write-Host "[dev-up] Done. Try:"
Write-Host "curl -i -H 'X-Tenant-Schema: $DevTenantSchema' -H 'X-Tenant-ID: dev-tenant' -H 'X-Case-ID: local' http://localhost:8000/ai/ping/"
