Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path '.env' -PathType Leaf)) {
    Write-Error '[dev-demo] Keine .env im Projektstamm gefunden. Bitte .env.example nach .env kopieren und Werte anpassen.'
}

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

Write-Host '[dev-demo] Building backend & job images'
Invoke-Expression "$compose --profile jobs build web worker migrate bootstrap"

Write-Host '[dev-demo] Bringing up core services'
Invoke-Expression "$compose up -d"

Write-Host '[dev-demo] Applying migrations & bootstrap'
Invoke-Expression 'npm run dev:init'

Write-Host '[dev-demo] Seeding demo tenant data'
Invoke-Expression 'npm run win:seed:demo'

Write-Host '[dev-demo] Fertig. Login unter http://demo.localhost:8000/admin/ mit demo/demo.'
Write-Host '[dev-demo] Hinweis: Hosts-Eintrag 127.0.0.1 demo.localhost erforderlich.'
