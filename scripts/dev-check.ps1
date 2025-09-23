Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$DevTenantSchema = if ($env:DEV_TENANT_SCHEMA) { $env:DEV_TENANT_SCHEMA } else { 'dev' }
$TenantId = if ($env:TENANT_ID) { $env:TENANT_ID } else { 'dev-tenant' }
$CaseId = if ($env:CASE_ID) { $env:CASE_ID } else { 'local' }

Write-Host "[dev-check] LiteLLM liveliness"
$ok = $false
for ($i=0; $i -lt 10; $i++) {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:4000/health/liveliness' -Method GET
    if ($resp.Content -match 'alive') { Write-Host 'LiteLLM alive'; $ok = $true; break }
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { Write-Warning 'LiteLLM liveliness failed' }

Write-Host "[dev-check] LiteLLM readiness"
try {
  $mk = $env:LITELLM_MASTER_KEY
  if ($mk) {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:4000/health' -Method GET -Headers @{ 'Authorization' = "Bearer $mk" }
    if ($resp.Content -match '"unhealthy_count":0') { Write-Host 'LiteLLM healthy' }
  } else {
    Write-Host 'Skipping readiness (no LITELLM_MASTER_KEY)'
  }
} catch { Write-Warning "LiteLLM readiness failed: $_" }

Write-Host "[dev-check] LiteLLM chat"
try {
  $mk = $env:LITELLM_MASTER_KEY
  if ($mk) {
    $body = @'
{
  "model": "gemini-2.5-flash",
  "messages": [{"role": "user", "content": "Sag \"ok\""}]
}
'@
    $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:4000/v1/chat/completions' -Method POST -ContentType 'application/json' -Headers @{ 'Authorization' = "Bearer $mk" } -Body $body
    if ($resp.Content -match '"choices"') { Write-Host 'Chat OK' } else { Write-Warning 'Chat response unexpected' }
  } else {
    Write-Host 'Skipping chat (no LITELLM_MASTER_KEY)'
  }
} catch { Write-Warning "LiteLLM chat failed: $_" }

Write-Host "[dev-check] AI Core ping with tenant headers"
$ok = $false
for ($i=0; $i -lt 5; $i++) {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/ai/ping/' -Method GET -Headers @{ 'X-Tenant-Schema' = $DevTenantSchema; 'X-Tenant-ID' = $TenantId; 'X-Case-ID' = $CaseId }
    Write-Host ("Status: {0}" -f $resp.StatusCode)
    $ok = $true; break
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { Write-Warning 'AI ping failed' }

Write-Host "[dev-check] POST /ai/scope minimal payload"
try {
  $payload = '{"hello":"world"}'
  $resp = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/ai/scope/' -Method POST -ContentType 'application/json' -Headers @{ 'X-Tenant-Schema' = $DevTenantSchema; 'X-Tenant-ID' = $TenantId; 'X-Case-ID' = $CaseId } -Body $payload
  $content = $resp.Content
  if ($content.Length -gt 200) { $content.Substring(0,200) + '...' } else { $content }
} catch { Write-Warning "AI scope failed: $_" }

try {
  Write-Host "[dev-check] RAG migrate"
  iex "docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T rag"
} catch { Write-Warning "RAG migrate failed: $_" }

try {
  Write-Host "[dev-check] RAG health"
  # Disable TTY to avoid interactive psql sessions on Windows
  iex "docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T rag-health"
} catch { Write-Warning "RAG health failed: $_" }

Write-Host "[dev-check] Done"
