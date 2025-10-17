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

Write-Host '[dev-demo] Demo-Tenant verwendet das In-Memory-Dokumentensystem'
Write-Host '[dev-demo] Hinweis: Dokument-Datasets werden nun über documents.cli gepflegt (siehe docs/documents/cli-howto.md).'
Write-Host '[dev-demo] Fertig. Demo-Tenant & Superuser jetzt via manage.py erzeugen:'
Write-Host '[dev-demo]   python manage.py create_tenant --schema=demo --name="Demo Tenant" --domain=demo.localhost'
Write-Host '[dev-demo]   python manage.py create_tenant_superuser --schema=demo --username=demo --email=demo@example.com'
Write-Host '[dev-demo] Hinweis: Hosts-Eintrag 127.0.0.1 demo.localhost erforderlich.'
