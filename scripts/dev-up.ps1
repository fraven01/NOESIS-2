Param(
    [string]$DevTenantSchema = $env:DEV_TENANT_SCHEMA,
    [string]$DevTenantName = $env:DEV_TENANT_NAME,
    [string]$DevDomain = $env:DEV_DOMAIN,
    [string]$DevSuperuserUsername = $env:DEV_SUPERUSER_USERNAME,
    [string]$DevSuperuserEmail = $env:DEV_SUPERUSER_EMAIL,
    [string]$DevSuperuserPassword = $env:DEV_SUPERUSER_PASSWORD,
    [switch]$IncludeElk,
    [switch]$SeedDemo,
    [switch]$SeedHeavy
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

$AppCompose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'
$ElkCompose = 'docker compose -f docker/elk/docker-compose.yml'

if ($IncludeElk -and (Test-Path '.env.dev-elk' -PathType Leaf)) {
    Write-Host '[dev-up] Lade .env.dev-elk (ELK Default-Credentials)'
    Get-Content '.env.dev-elk' | ForEach-Object {
        $line = $_
        if (-not [string]::IsNullOrWhiteSpace($line)) {
            $trimmed = $line.Trim()
            if (-not $trimmed.StartsWith('#') -and $trimmed.Contains('=')) {
                $parts = $trimmed -split '=', 2
                if ($parts.Length -eq 2) {
                    $key = $parts[0].Trim()
                    $value = $parts[1].Trim()
                    $value = $value.Trim('"')
                    $value = $value.Trim([char]39)
                    if ($key) {
                        Set-Item -Path "Env:$key" -Value $value
                        [System.Environment]::SetEnvironmentVariable($key, $value)
                    }
                }
            }
        }
    }
}

if ($IncludeElk) {
    # Try to ensure vm.max_map_count is set for Elasticsearch on Docker Desktop's Linux VM
    try {
        Write-Host '[dev-up] Ensuring vm.max_map_count=262144 in docker-desktop WSL VM'
        & wsl -d docker-desktop -u root sh -lc "sysctl -w vm.max_map_count=262144" | Out-Null
    } catch {
        Write-Host '[dev-up] Could not set vm.max_map_count automatically (continuing)'
    }
    $AppLogPath = $env:APP_LOG_PATH
    if (-not $AppLogPath) {
        $AppLogPath = Join-Path -Path (Get-Location) -ChildPath 'logs/app'
    } elseif (-not [System.IO.Path]::IsPathRooted($AppLogPath)) {
        $relativeLogPath = $AppLogPath.TrimStart('.', '/', [char]92)
        if (-not $relativeLogPath) {
            $relativeLogPath = $AppLogPath
        }
        $AppLogPath = Join-Path -Path (Get-Location) -ChildPath $relativeLogPath
    }
    $env:APP_LOG_PATH = $AppLogPath
    $env:APP_LOG_DIR = $AppLogPath

    if (-not (Test-Path -Path $AppLogPath)) {
        New-Item -ItemType Directory -Path $AppLogPath | Out-Null
    }
    Write-Host "[dev-up] Log-Verzeichnis: $AppLogPath"

    # Ensure env compatibility: if only KIBANA_SYSTEM_PASSWORD is set, mirror to KIBANA_PASSWORD
    if (-not $env:KIBANA_PASSWORD -and $env:KIBANA_SYSTEM_PASSWORD) {
        $env:KIBANA_PASSWORD = $env:KIBANA_SYSTEM_PASSWORD
        Write-Host '[dev-up] Using KIBANA_SYSTEM_PASSWORD as KIBANA_PASSWORD for Kibana/Elasticsearch'
    }

    Write-Host '[dev-up] Building application stack images'
    Invoke-Expression "$AppCompose build"

    Write-Host '[dev-up] Building ELK stack images'
    Invoke-Expression "$ElkCompose build"
}

Write-Host "[dev-up] Bringing up services (db, redis, litellm, web, worker)"
Invoke-Expression "$AppCompose up -d"

if ($IncludeElk) {
    # Start Elasticsearch first to control password bootstrap order
    Write-Host '[dev-up] Starting Elasticsearch (ELK)'
    Invoke-Expression "$ElkCompose up -d elasticsearch"

    # Wait until Elasticsearch is healthy
    $esOk = $false
    $esPassword = if ($env:ELASTIC_PASSWORD) { $env:ELASTIC_PASSWORD } else { 'changeme' }
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $pair = "elastic:$esPassword"
            $basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
            $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:9200' -Headers @{ Authorization = "Basic $basic" } -TimeoutSec 3
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { $esOk = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $esOk) { Write-Host '[dev-up] Elasticsearch not ready yet, continuing but Kibana may fail to auth' }

    # Ensure kibana_system password matches KIBANA_PASSWORD (idempotent) via REST API
    $kibanaPwd = if ($env:KIBANA_PASSWORD) { $env:KIBANA_PASSWORD } else { 'changeme' }
    try {
        Write-Host '[dev-up] Ensuring kibana_system password in Elasticsearch via REST API'
        $pair = "elastic:$esPassword"
        $basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
        $headers = @{ Authorization = "Basic $basic"; 'Content-Type' = 'application/json' }
        $body = @{ password = $kibanaPwd } | ConvertTo-Json -Compress
        Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:9200/_security/user/kibana_system/_password' -Method Post -Headers $headers -Body $body -TimeoutSec 5 | Out-Null
    } catch {
        Write-Host '[dev-up] Could not set kibana_system password via API (continuing)'
    }

    # Start Kibana and Logstash after password is ensured
    Write-Host '[dev-up] Starting Kibana and Logstash'
    Invoke-Expression "$ElkCompose up -d kibana logstash"
}

Write-Host "[dev-up] Waiting for web to respond (warm-up)"
$ok = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/' -Method GET
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
            Write-Host "[dev-up] Web responded with HTTP $($resp.StatusCode)"
            $ok = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ok) {
    Write-Host '[dev-up] Web not responding yet, continuing'
}

Write-Host "[dev-up] Init jobs: migrate + bootstrap"
Invoke-Expression 'npm run dev:init'

if ($IncludeElk -or $SeedDemo) {
    Write-Host '[dev-up] Seeding demo tenant dataset'
    Invoke-Expression "$AppCompose exec web python manage.py create_demo_data --profile demo --seed 1337"
}

if ($IncludeElk -or $SeedHeavy) {
    Write-Host '[dev-up] Seeding heavy dataset'
    Invoke-Expression "$AppCompose exec web python manage.py create_demo_data --profile heavy --seed 42"
}

$tenantSchema = if ($env:DEV_TENANT_SCHEMA) { $env:DEV_TENANT_SCHEMA } else { 'demo' }
$tenantId = if ($env:DEV_TENANT_ID) { $env:DEV_TENANT_ID } else { 'demo' }
Write-Host '[dev-up] Optional tenant ping (nach Bootstrap):'
$tenantPing = @"
curl -i `
  -H 'X-Tenant-Schema: $tenantSchema' `
  -H 'X-Tenant-Id: $tenantId' `
  -H 'X-Case-ID: local' `
  http://localhost:8000/ai/ping/
"@
Write-Host $tenantPing

Write-Host '[dev-up] Done. Optional tenant ping (einzeilig):'
Write-Host "curl -i -H 'X-Tenant-Schema: $tenantSchema' -H 'X-Tenant-Id: $tenantId' -H 'X-Case-ID: local' http://localhost:8000/ai/ping/"

if ($IncludeElk) {
    Write-Host '[dev-up] Kibana läuft unter http://localhost:5601 (ELASTIC_PASSWORD erforderlich).'
}
