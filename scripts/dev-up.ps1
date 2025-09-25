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

    if (-not (Test-Path -Path $AppLogPath)) {
        New-Item -ItemType Directory -Path $AppLogPath | Out-Null
    }
    Write-Host "[dev-up] Log-Verzeichnis: $AppLogPath"

    Write-Host '[dev-up] Building application stack images'
    Invoke-Expression "$AppCompose build"

    Write-Host '[dev-up] Building ELK stack images'
    Invoke-Expression "$ElkCompose build"
}

Write-Host "[dev-up] Bringing up services (db, redis, litellm, web, worker)"
Invoke-Expression "$AppCompose up -d"

if ($IncludeElk) {
    Write-Host '[dev-up] Starting ELK stack'
    Invoke-Expression "$ElkCompose up -d"
}

Write-Host "[dev-up] Waiting for web to respond (warm-up)"
$ok = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/ai/ping/' -Method GET
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
    Invoke-Expression 'npm run seed:demo'
}

if ($IncludeElk -or $SeedHeavy) {
    Write-Host '[dev-up] Seeding heavy dataset'
    Invoke-Expression 'npm run seed:heavy'
}

Write-Host "[dev-up] Done. Try:"
Write-Host "curl -i -H 'X-Tenant-Schema: $DevTenantSchema' -H 'X-Tenant-ID: dev-tenant' -H 'X-Case-ID: local' http://localhost:8000/ai/ping/'"

if ($IncludeElk) {
    Write-Host '[dev-up] Kibana läuft unter http://localhost:5601 (ELASTIC_PASSWORD erforderlich).'
}
