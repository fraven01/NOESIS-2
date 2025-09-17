Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$envFile = '.env'
if (-not (Test-Path $envFile)) {
  Write-Error '[rag-enable] .env not found'
  exit 1
}

$content = Get-Content $envFile -Raw
if ($content -match "(?m)^RAG_ENABLED=") {
  $new = [regex]::Replace($content, "(?m)^RAG_ENABLED=.*$", 'RAG_ENABLED=true')
} else {
  $newline = if ($content.EndsWith("`n")) { '' } else { "`n" }
  $new = $content + $newline + 'RAG_ENABLED=true' + "`n"
}
Set-Content $envFile -Value $new -NoNewline:$false
Write-Host '[rag-enable] Set RAG_ENABLED=true in .env'

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'
Write-Host '[rag-enable] Restarting web and worker'
iex "$compose restart web worker"
Write-Host '[rag-enable] Done'

