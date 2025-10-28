# Warum
Die Pipeline sorgt dafür, dass jede Änderung getestet, geprüft und kontrolliert ausgerollt wird. Diese Anleitung zeigt die feste Reihenfolge der Stufen und die benötigten Berechtigungen.

# Wie
## Pipeline-Stufen
| Reihenfolge | Stufe | Inhalt | Gate |
| --- | --- | --- | --- |
| 1 | Lint | `npm run lint` inklusive Ruff/Black, siehe `.github/workflows/ci.yml` | Merge blockiert bei Fehlern |
| 2 | Unit | `pytest` mit Tenant-Migrationen und Coverage (inkl. `ai_core/tests/test_crawler_*.py`, siehe `.github/workflows/ci.yml`) | Must-pass für Image-Build |
| 3 | Build | Docker Build mit Multi-Stage, Tag `vX.Y.Z-<sha>` (Semver+SHA) | Nutzt Artifact Registry Service Account |
| 4 | Push | Docker Push zum Artifact Registry Repo | Abbruch bei fehlender Auth |
| 5 | E2E (Compose) | Playwright Tests aus `.github/workflows/e2e.yml` gegen Docker-Compose-Stack, prüft Retrieval-Ende-zu-Ende | Ergebnis entscheidet über Staging-Deploy |
| 6 | Vector-Schema-Migrations | Führt SQL aus [`docs/rag/schema.sql`](../rag/schema.sql) über Cloud SQL Proxy; dry-run + Apply, Laufzeit derzeit ohne Bewertung | Stoppt Release bei DDL-Fehlern |
| 7 | Staging Migrate Job | Cloud Run Job `noesis2-migrate` mit `migrate_schemas --noinput` | Bricht Release bei Fehlern ab |
| 8 | Staging Deploy | Cloud Run Deploy für Web, Worker, Beat, LiteLLM mit `--set-env-vars` | Löst Smoke-Tests aus |
| 9 | Ingestion-Smoke | Cloud Run Job (Queue `ingestion`) verarbeitet Mini-Batch, verifiziert Embeddings; keine Backfill-Jobs für Altbestände | Blockiert Approval bei Fehler |
| 10 | Smoke Staging | Health-Checks (`/`, `/health/liveliness`) + Logsichtung laut [QA](../qa/checklists.md) | Gate für Approval |
| 10a | Chaos Smoke (optional) | Manuell via `tests-chaos` Workflow-Dispatch (`run_chaos=true`), führt `pytest -m chaos -q -n auto` aus und lädt JUnit/JSON/k6/Locust-Artefakte hoch | QA-Abbruchkriterien: Fehlerquote <5 %, Latenz-P95 unter Grenzwert laut [QA-Checklisten](../qa/checklists.md) |
| 11 | Approval | Manuelle Freigabe (Prod Approver) in GitHub Actions | Pflicht vor Prod |
| 12 | Prod Migrate | Cloud Run Job mit Secrets aus Secret Manager | Stoppt Flow bei Fehler |
| 13 | Prod Deploy (Traffic-Split) | Deploy neue Revision mit z.B. 10% Traffic | Startet Prod-Smoke |
| 14 | Smoke Prod | HTTP-Checks, Log-Review, Monitoring laut [QA](../qa/checklists.md) | Vor Bedarfs-Traffic |
| 15 | Traffic 100% | Erhöht Anteil auf 100%, bestätigt in GitHub Actions | Markiert Release als abgeschlossen |

## Artefakt-Tagging
- Versionen folgen Semver auf Branch-Ebene. Major/Minor/Patch wird bei Release-Tags gesetzt, `-<commit-sha>` stellt Eindeutigkeit sicher.
- Jedes Tag landet als Cloud Run Revision Label (`version`, `commit`) für Nachvollziehbarkeit.
- Rollbacks verwenden den letzten Tag aus der Registry; Deployments ohne Tag sind nicht erlaubt.

## Workload Identity Federation
Ein gemeinsames Service-Konto interagiert mit GCP über WIF. Es erhält ausschließlich die minimal nötigen Rollen:
- `roles/artifactregistry.writer` für Build & Push
- `roles/run.admin` und `roles/run.developer` für Deployments
- `roles/cloudsql.client` für Migrate-Jobs
- `roles/cloudsql.editor` oder granularer DDL-Rolle für Vector-Schema-Migrations
- `roles/secretmanager.secretAccessor` nur in Prod-Stufen
- `roles/redis.admin` für Memorystore-Verbindungen
- `roles/compute.networkUser` damit Serverless VPC Access genutzt werden darf

# Schritte
1. Pflege Semver-Versionen vor dem Merge, damit Build- und Deploy-Stufen das korrekte Tag erzeugen.
2. Stelle sicher, dass Unit- und E2E-Tests vollständig laufen und Ergebnisse dokumentiert sind.
3. Überprüfe vor jedem Release, ob WIF-Rollen für das Pipeline-Service-Konto aktuell sind.
4. Folge dem Gate-Flow strikt: kein Prod-Deploy ohne bestandenes Staging und manuelle Freigabe.
5. Dokumentiere Smoke-Ergebnisse und Traffic-Entscheidungen in den Release-Notizen.
6. Optional: Nach bestandenem Staging-Smoke-Test löse den Workflow `CI` via `workflow_dispatch` mit `run_chaos=true` aus, analysiere die Artefakte des Jobs `tests-chaos` (JUnit, JSON, k6/Locust) und vergleiche Fehlerquote sowie P95-Latenz mit den Abbruchkriterien der [QA-Checklisten](../qa/checklists.md).
