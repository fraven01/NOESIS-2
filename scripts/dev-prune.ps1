Param(
    [switch]$All
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host '[dev-prune] Pruning dangling images and builder cache'
docker image prune -f | Out-Null
docker builder prune -f | Out-Null

if ($All) {
    Write-Host '[dev-prune] Also pruning unused networks and volumes (destructive)'
    docker network prune -f | Out-Null
    docker volume prune -f | Out-Null
}

Write-Host '[dev-prune] Done'

