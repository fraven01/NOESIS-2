# Warum
Sichere Geheimnisse verhindern Datenverlust und Ausfälle. Dieses Dokument definiert die ENV-Verträge und beschreibt, wie sie in den Umgebungen verwaltet werden.

# Wie
## ENV-Verträge
| Variable | Beschreibung | Dev | Staging | Prod |
| --- | --- | --- | --- | --- |
| `DATABASE_URL` | Postgres-Verbindungsstring, von Django und LiteLLM genutzt | `.env` lokal, häufig identisch mit `postgres://user:pass@db:5432/noesis2` | Aus CI gesetzt (`--set-env-vars`), zeigt auf Cloud SQL Public IP | Secret Manager, Version pro Rotation; verweist auf Cloud SQL Private IP |
| `REDIS_URL` | Redis-Broker für Celery und Caching | `.env` → `redis://redis:6379/0` | CI setzt Wert auf Memorystore Instanz (`rediss://`) | Secret Manager → Memorystore (TLS, Auth-Token) |
| `DJANGO_SETTINGS_MODULE` | Django Settings Modul | Lokal `noesis2.settings.development` | CI setzt `noesis2.settings.production` | Secret Manager oder Runtime Config → `noesis2.settings.production` |
| `SECRET_KEY` | Django Crypto Key | `.env` zufällig generiert | CI Secret (per GitHub Secret) | Secret Manager Version, Rotation 90 Tage |
| `ALLOWED_HOSTS` | Kommagetrennte Domains | `.env` (`localhost`) | CI-Variable mit Staging-Domain | Secret Manager, Prod-Domains + LB |
| `LITELLM_URL` | Basis-URL für LiteLLM Proxy (Alias für `LITELLM_BASE_URL` im Code) | `.env` zeigt auf `http://litellm:4000` | CI setzt auf interne Cloud Run URL, nur auth Nutzer | Secret Manager liefert interne URL; optional zweiter Eintrag für IAP |
| `EMBEDDINGS_MODEL` | Modellname für Embedding-Erstellung | `.env` frei wählbar (z.B. `text-embedding-3-small`) | CI-Variable pro Deploy; Versionswechsel nur nach Smoke-Test | Secret Manager Version, dokumentiert in Release-Notes |
| `EMBEDDINGS_DIM` | Dimensionszahl der Vektoren (Integer) | `.env` konsistent zur lokalen Modellwahl | CI setzt Wert passend zum Modell | Secret Manager, gemeinsam mit Modellwechsel aktualisieren |
| `EMBEDDINGS_API_BASE` | LiteLLM-Endpunkt für Embedding-Requests | `.env` → `http://litellm:4000/v1` | CI `--set-env-vars` nutzt Cloud Run interne URL | Secret Manager verweist auf HTTPS Load Balancer oder interne URL |
| `LANGFUSE_HOST` | Basis-URL für Langfuse Workspace | Optional lokal (`http://localhost:3000`) | CI-Variable; SaaS oder Self-Host URL | Secret Manager Wert, kein direkter Konsolenzugriff |
| `LANGFUSE_KEY` | Server-SDK-Key für Worker und Agenten | `.env` Dummy | CI-Secret | Secret Manager Version, Rotation 60 Tage |
| `LANGFUSE_PUBLIC_KEY` | Browser-Key für LiteLLM GUI Trace | Nicht gesetzt | CI-Secret | Secret Manager Version |
| `LANGFUSE_SECRET_KEY` | Privater Key für LiteLLM Trace Upload | Nicht gesetzt | CI-Secret | Secret Manager Version |

Weitere Schlüssel wie `GEMINI_API_KEY`, `LITELLM_MASTER_KEY`, `AI_CORE_RATE_LIMIT_QUOTA` folgen denselben Quellen und werden nicht im Image gespeichert.

## PII-Redaction und Log-Scopes
- Web- und Worker-Services setzen `LOG_SCOPE=tenant` und entfernen personenbezogene Daten vor Log-Schreibungen. Maskierung erfolgt über die Middleware aus `ai_core.pii`.
- Langfuse nutzt `PII_REDACTION_RULES` (Regex oder Hashing) zur Entfernung von Namen, E-Mail-Adressen und Vertragsnummern; Regeln werden in Secret Manager abgelegt.
- LiteLLM zeichnet Prompts nur anonymisiert auf (`store_prompt=false` für sensible Modelle) und sendet Trace-Metadaten an Langfuse.
- Zugriff auf Logs: Entwickler lesen Staging-Logs, Prod nur Observability-Team mit Audit-Log.

## Rotation
- Rotationen erfolgen mindestens alle 90 Tage bzw. vor Freigabe sensibler Features.
- Dev: Entwickler erstellen neue `.env`-Werte manuell; keine Weitergabe via Git.
- Staging: GitHub Secret aktualisieren, Pipeline triggert Redeploy, alte Values werden invalidiert.
- Prod: Neue Secret-Version in Secret Manager anlegen, Revision deployen, alte Version nach Validierung deaktivieren.
- LiteLLM Keys rotieren paarweise (Master + API). Informiere alle Dienste, die `LITELLM_URL` nutzen, und teste `GET /health` danach.

## Kein Secret im Image
Das [`Dockerfile`](../../Dockerfile) kopiert nur Quellcode und generierte Assets. Secrets gelangen ausschließlich zur Laufzeit über Umgebungsvariablen. Prüfe Pull Requests darauf, dass keine Credentials eingecheckt werden.

# Schritte
1. Definiere neue Secrets immer zuerst in diesem Dokument und stimme sie mit dem Team ab.
2. Lege die Werte gemäß Tabelle in der richtigen Quelle (Dev `.env`, CI Secret, Secret Manager) ab.
3. Redeploye betroffene Dienste und prüfe Health-Checks sowie Logs.
4. Entferne alte Secret-Versionen nach erfolgreichem Rollout und dokumentiere die Rotation.
