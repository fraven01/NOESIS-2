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

## Ingestion Smoke (Pipeline Stufe 9)
- Verifiziere, dass der Cloud Run Job für die Queue `ingestion` den letzten Mini-Batch erfolgreich verarbeitet hat (`Job completed successfully`).
- Prüfe in Langfuse, dass der Trace `retrieval_results>0` liefert und keine Embedding-Rate-Limits auftreten (siehe [Incidents-Runbook](../runbooks/incidents.md)).
- Stelle sicher, dass `ingestion`-Queues nach dem Lauf leer sind und keine Retries offen bleiben.
- Dokumentiere den Batch (Tenant, Dokumentanzahl, Timestamp) im Release-Template.

## Staging Smoke-Test (Pipeline Stufe 10)
- Führe HTTP-Checks auf `/`, `/health/liveliness` und `/tenant-demo/` durch; alle Antworten müssen `200` liefern (vgl. [Migrations-Runbook](../runbooks/migrations.md)).
- Überprüfe Cloud Logging auf neue Fehlerstapel seit dem Deploy und bestätige, dass die Error-Rate unter 5 % bleibt.
- Kontrolliere Celery-Queues und Redis-Keys auf Stabilität; LiteLLM `/health` darf keine Fehlerzählung anzeigen.
- Vergleiche Schema-Versionen der `django_migrations` Tabelle mit dem erwarteten Stand aus dem Release.

## Chaos Smoke (optional nach Staging)
- Starte den GitHub Actions Workflow `CI` manuell mit `run_chaos=true`, sobald die Stufen 8–10 erfolgreich waren.
- Prüfe im Artefakt `chaos-test-reports` die JUnit- und JSON-Ausgabe: Fehlerquote muss <5 % bleiben, P95-Latenz darf die Grenzwerte aus den Chaos-Smoke-Kennzahlen nicht überschreiten.
- Lade die optionalen k6-/Locust-Summaries (`chaos-load-summaries`) herunter und dokumentiere Spike-/Soak-Raten, Latenz-P95 sowie Trefferquote.
- Bei Verletzung der Schwellenwerte: Release stoppen, Incident-Runbook öffnen und Chaos-Toggles deaktivieren, bevor erneute Deployments erfolgen.

## Prod Smoke & Traffic-Freigabe (Pipeline Stufen 14–15)
- Wiederhole alle Staging-Smoke-Prüfungen in Prod unmittelbar nach dem Traffic-Split.
- Beobachte Cloud Monitoring Metriken (Latenz, Fehler, Queue-Tiefe) engmaschig und vergleiche sie mit den letzten 24 Stunden.
- Bei Auffälligkeiten: Stoppe Traffic-Erhöhungen und folge den Eskalationspfaden im [Incidents-Runbook](../runbooks/incidents.md).
- Dokumentiere die Entscheidung über die Traffic-Erhöhung auf 100 % in den Release-Notizen.

## Rollen & Dokumentation
- **Release Manager** verantwortet die Durchführung der QA-Checks und die Freigabeentscheidungen laut [Pipeline](../cicd/pipeline.md).
- **On-Call/Incident Commander** wird eingebunden, sobald ein Abbruchkriterium eintritt oder Eskalation gemäß Runbook nötig ist.
- Alle Prüfergebnisse, Logs und Entscheidungen werden im Projekt-Wiki archiviert; Abweichungen fließen in die Aktualisierung der Runbooks ein.
