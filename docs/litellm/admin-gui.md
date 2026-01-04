# Warum
LiteLLM bündelt alle LLM-Zugriffe und liefert eine Admin-GUI. Dieses Dokument erklärt Betrieb, Berechtigungen und Grenzen, damit der Dienst kontrolliert bleibt.

# Wie
## Betriebsmodell
- LiteLLM läuft als eigener Cloud Run Service (siehe [Architektur](../architecture/overview.md)).
- Konfiguration stammt aus [`config/litellm-config.yaml`](../../config/litellm-config.yaml); `require_auth: true` erzwingt Authentifizierung.
- Der Dienst verwendet `DATABASE_URL`, um Nutzungsdaten und API-Keys zu speichern. In Cloud-Umgebungen liegt diese Datenbank im selben Cloud SQL Projekt wie Django.

## Berechtigungen
- Zugriffe erfolgen über API Keys (Master Key für Admins, individuelle Keys für Dienste). Master Key wird wie in [Security](../security/secrets.md) verwaltet.
- Cloud Run Auth (`roles/run.invoker`) beschränkt HTTP-Zugriffe. In Prod optional zusätzlich Cloud IAP für die GUI.
- Änderungen an Modelleinstellungen erfolgen nur durch AI Platform Owner oder delegierte Admins.

## Rate-Limits & Protokollierung
- `AI_CORE_RATE_LIMIT_QUOTA` steuert die erlaubten Requests pro Tenant. Anpassungen passieren synchron in Django und LiteLLM, damit Limits übereinstimmen.
- LiteLLM protokolliert alle Requests in Cloud Logging und Langfuse (ENV `LANGFUSE_*`). Logs werden 30 Tage (Staging) bzw. 90 Tage (Prod) aufbewahrt.
- „Modell-Trace aktivieren“: Setze `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` und `LANGFUSE_KEY` im LiteLLM-Service. Optional `LANGFUSE_SAMPLE_RATE` für feinere Steuerung, siehe [Langfuse Observability](../observability/langfuse.md).
- Rate-Limits pro Team: Weise jedem API-Key einen Tenant zu und setze `rate_limit_per_minute` im LiteLLM Config Mapping.
- Fehler und Überschreitungen lösen Alerts aus, die mit dem [Incident-Runbook](../runbooks/incidents.md) verknüpft sind.

Zusätzliches GUI-Monitoring oder gesondertes Logging ist nicht erforderlich; Standard-Observability reicht.

## Risiken
- Fehlkonfigurierte Keys können unbegrenzte Kosten erzeugen.
- Ohne Auth könnten sensible Prompts abfließen.
- Datenbankdrift zwischen Django und LiteLLM führt zu Inkonsistenzen bei Limits oder Billing.

## Do & Don't
- **Do**: Keys regelmäßig rotieren, Logs prüfen, GUI nur über abgesicherte Wege öffnen.
- **Do**: Deployments synchron mit App-Release planen, damit `DATABASE_URL` und Migrationsstand passen.
- **Don't**: GUI öffentlich freischalten oder Keys per Chat-Tools teilen.
- **Don't**: LiteLLM-Config direkt in der Cloud Run Konsole ändern; Änderungen gehören ins Repo.

## Zugriffsmuster
- **Staging**: Zugriff über interne Cloud Run URL + Identity-Aware Proxy optional; QA nutzt eigene API Keys mit begrenzten Limits.
- **Prod**: Zugriff ausschließlich über HTTPS Load Balancer bzw. IAP. Nur freigegebene Service-Konten oder Benutzergruppen dürfen Requests senden.
- Modell- und Trace-Zugriff: Nur Observability-Team darf Langfuse Credentials für Prod einsehen; jede Änderung wird protokolliert.

# Schritte
1. Verwalte Keys gemäß [Security](../security/secrets.md) und verteile neue Werte nur über sichere Kanäle.
2. Prüfe nach Deployments die LiteLLM Health-Endpunkte, Langfuse-Traces und Log-Dashboards.
3. Kontrolliere regelmäßig Rate-Limit-Reports und Tenant-Kosten; setze Alerts in Langfuse und Cloud Monitoring.
4. Dokumentiere jede Konfigurationsänderung in der Admin-GUI im Änderungsprotokoll und verweise auf Langfuse Dashboards.
