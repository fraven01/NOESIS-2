Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

git config core.hooksPath .githooks | Out-Null
try {
  & bash -lc "chmod +x .githooks/pre-push" | Out-Null
} catch {}
Write-Host 'Git hooks installed (core.hooksPath=.githooks)'

