# Hybrid-RAG-Demo von Grund auf einrichten

Dieser Plan führt von einer leeren lokalen Umgebung („alles neu, inklusive Datenbank“)
zu einer funktionsfähigen Demo, die den Upload eines Testdokuments, die
Ingestion-Pipeline und die hybride Suche über den `rag-demo`-Endpoint abdeckt. Er
orientiert sich an den Vorgaben aus [`docs/rag/ingestion.md`](ingestion.md), der
API-Referenz und den vorhandenen Dev-Skripten.

## 1. Voraussetzungen schaffen
- **Tooling**: `docker`, `docker compose`, `npm`, `curl`, `jq`, `psql`.
- **Repo vorbereiten**: `cp .env.example .env` und alle benötigten Secrets/Keys
  gemäß README eintragen (u. a. Embeddings-Provider für pgvector).
- **Hosts-Datei**: `127.0.0.1 demo.localhost` eintragen, damit der Demo-Hostname
  auf die lokale Maschine zeigt.

## 2. Umgebung & Datenbank neu aufsetzen
1. Alte Container und Volumes entfernen:
   ```bash
   ./scripts/dev-reset.sh
   ```
   Das Skript baut Images neu, startet Web/Worker/DB und führt `npm run dev:init`
   (Migrationen, Tenants, Seeds) sowie `npm run dev:check` aus.
2. Prüfen, dass die Dienste laufen:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
   docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f worker
   ```
   Der Worker muss die Queue `ingestion` horchen; auftretende Fehler sofort
   beseitigen.
3. Sicherstellen, dass das RAG-Schema vorhanden ist:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml exec postgres \
     psql -U noesis -d noesis -c "\dn" | grep rag
   ```
   Fehlt das Schema, `docs/rag/schema.sql` gegen die Datenbank ausführen.

## 3. Testdatei vorbereiten
1. `hello.txt` mit eindeutigem Inhalt erzeugen (Beispiel):
   ```bash
   cat <<'TXT' > hello.txt
   Hallo ZEBRAGURKE,
   dies ist ein End-to-End-Test für die RAG-Demo.
   TXT
   ```
2. Externe ID für die Nachverfolgung festlegen, z. B. `demo-hello-$(date +%s)`.

## 4. Dokument hochladen
1. Upload per `curl` ausführen:
   ```bash
   EXTERNAL_ID="demo-hello-$(date +%s)"
   curl -sS -X POST "http://demo.localhost:8000/ai/rag/documents/upload/" \
     -H "X-Tenant-Schema: demo" \
     -H "X-Tenant-Id: demo" \
     -H "X-Case-Id: local" \
     -F "file=@hello.txt" \
     -F "metadata={\"external_id\":\"$EXTERNAL_ID\",\"label\":\"smoke\"}" \
     | tee /tmp/upload.json | jq .
   ```
2. Erwartetes Ergebnis: HTTP `202`, Felder `document_id`, `external_id`, `trace_id`.
   `document_id` für den nächsten Schritt notieren.

## 5. Ingestion starten und überwachen
1. Ingestion-Run anstoßen:
   ```bash
   DOCUMENT_ID=$(jq -r '.document_id' /tmp/upload.json)
   curl -sS -X POST "http://demo.localhost:8000/ai/rag/ingestion/run/" \
     -H "Content-Type: application/json" \
     -H "X-Tenant-Schema: demo" \
     -H "X-Tenant-Id: demo" \
     -H "X-Case-Id: local" \
     -d "{\"document_ids\":[\"$DOCUMENT_ID\"],\"priority\":\"normal\"}" \
     | tee /tmp/ingestion.json | jq .
   ```
   Antwort: HTTP `202` mit `ingestion_run_id` und optional `invalid_ids`.
2. Worker-Logs beobachten, bis `ingestion_run` abgeschlossen ist:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml \
     logs -f --tail=100 worker | grep -E "ingestion_run|process_document"
   ```
   Erfolgsindikator: Meldungen wie `written=...` ohne Fehlerstacktrace.
3. Datenbankprüfung (optional, aber empfohlen):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml exec postgres \
     psql -U noesis -d noesis <<'SQL'
   SET search_path TO rag, public;
   SELECT id, external_id, metadata FROM documents ORDER BY created_at DESC LIMIT 5;
   SELECT document_id, ord, LEFT(text, 60) FROM chunks ORDER BY created_at DESC LIMIT 5;
   SELECT COUNT(*) FROM embeddings;
   SQL
   ```
   Die neue `external_id` muss erscheinen, `chunks`/`embeddings` dürfen nicht 0 sein.

## 6. Hybride Suche validieren
1. Anfrage mit bewusst gesetzten Parametern schicken:
   ```bash
   curl -sS -X POST "http://demo.localhost:8000/ai/v1/rag-demo/" \
     -H "Content-Type: application/json" \
     -H "X-Tenant-Schema: demo" \
     -H "X-Tenant-Id: demo" \
     -H "X-Case-Id: local" \
     -d '{
       "query": "ZEBRAGURKE",
       "top_k": 5,
       "alpha": 0.6,
       "min_sim": 0.15,
       "vec_limit": 20,
       "lex_limit": 30,
       "trgm_limit": 0.30
     }' | tee /tmp/hybrid.json | jq .
   ```
2. Prüfpunkte:
   - `matches` enthält das neue Dokument (`metadata.external_id == $EXTERNAL_ID`).
   - Feld `error` fehlt; `meta.alpha` und `meta.min_sim` entsprechen den gesetzten
     Werten.
   - `meta.vector_candidates`/`meta.lexical_candidates` > 0 (Beweis für Hybrid).
   - `score` plausibel (> 0) und nicht der Demo-Fallbackwert `0.42`/`0.36`.
3. Variationstests: `alpha=0.0` (nur lexical) und `alpha=1.0` (nur vector) senden
   und Score-/Kandidatenverhalten vergleichen.

## 7. Aufräumen
- Temporäre JSONs (`/tmp/upload.json`, `/tmp/ingestion.json`, `/tmp/hybrid.json`)
  löschen.
- Optional: `docker compose ... down -v` zum Zurücksetzen.

## 8. Automatisierung
Das frühere Skript [`scripts/rag_demo_walkthrough.sh`](../../scripts/rag_demo_walkthrough.sh)
wurde im MVP als veraltet markiert und bricht sofort ab. Die automatisierte Demo
ist damit deaktiviert; nutze stattdessen die produktiven RAG-Flows.
