# Warum
Qualitätssicherung verhindert Überraschungen nach Deployments. Diese Checklisten beschreiben, was vor und nach jedem Rollout geprüft wird.

# Wie
## Vor dem Deploy
- Build-Artefakte sind vorhanden (Semver+SHA Tag in Artifact Registry) und wurden signiert, falls erforderlich.
- Migrationen: Trockenlauf per Cloud Run Job in Staging durchgeführt, Ergebnis dokumentiert ([Migrations-Runbook](../runbooks/migrations.md)).
- Konnektivität: Smoke-Verbindungen zu Cloud SQL und Memorystore geprüft; LiteLLM `/health/liveliness` liefert `alive`.
- Monitoring: Alerts für CPU, Fehlerquote und Queue-Länge aktiv und ohne offene Warnungen.

## Nach dem Deploy
- `/` und `/admin/login/` antworten mit HTTP 200, LiteLLM `/health` meldet keine Fehlerzählung.
- Cloud Logging zeigt keine neuen Fehlerstapel; Error Reporting frei von kritischen Einträgen.
- Metriken (Latenz, Fehler, Queue-Tiefe) bleiben im grünen Bereich der letzten 24 Stunden.
- Backups aktiv: Cloud SQL Backupstatus `SUCCESS`, PITR-Zeitfenster vorhanden, Memorystore Snapshot aktualisiert.

## Abbruchkriterien & Rollback
- Smoke-Test fehlschlägt oder Fehlerquote >5%: sofort Traffic auf vorherige Revision (siehe [Pipeline](../cicd/pipeline.md)).
- Migration schlägt fehl: kein weiterer Deploy, Datenbankzustand sichern und [Runbook](../runbooks/migrations.md) folgen.
- LiteLLM oder Worker nicht verfügbar: Incidents gemäß [Runbook](../runbooks/incidents.md) starten, Release pausieren.

# Schritte
1. Führe alle Vorab-Checks aus und dokumentiere Ergebnisse im Release-Template.
2. Nach Deploy: Arbeite die Nach-Deploy-Liste ab und dokumentiere Zeitpunkte.
3. Entscheide anhand der Abbruchkriterien, ob Traffic erhöht, eingefroren oder zurückgerollt wird.
4. Hinterlege alle Ergebnisse im Projekt-Wiki, damit Lessons Learned nachvollziehbar sind.

## Wird nach Implementierung ergänzt
