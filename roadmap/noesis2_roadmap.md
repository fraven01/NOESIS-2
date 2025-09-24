# NOESIS-2 Verbesserungs-Roadmap

Diese Roadmap bewertet den aktuellen Stand der NOESIS-2 Plattform und zeigt Verbesserungsbedarf auf, um auf „State of the Art“-Niveau zu kommen.

---

## Stärken
- **Mandantenfähigkeit**: sauber mit `django-tenants` und `X-Tenant-Schema`-Header gelöst.
- **Idempotenz**: Kopfzeilen + Registry verhindern Doppelverarbeitungen.
- **Ingestion**: dedupliziert, Backoff, Dead-Letter-Queue – solide Grundlage.
- **PII-Scope**: deterministische Redaction, stabilisiert durch HMAC.
- **Pipeline & Ops**: Gates, Smoke-Checks, Canary-Deploys.
- **Observability**: Traces über Langfuse + ELK.

Das ist bereits über Industriestandard hinaus.

---

## Verbesserungsbedarf

### 1. API-Contract-First
- Aktuell: Dokumentation vorhanden, aber nachträglich gepflegt.
- Verbesserung: OpenAPI-Spec direkt aus Code generieren (z. B. drf-spectacular).
- Nutzen: CI-Validation, SDK-Generierung, API-first Entwicklung.

### 2. Security by Design
- Aktuell: Secrets sauber gelöst, API-Key Auth.
- Verbesserung: 
  - Zero Trust / mTLS zwischen Services.
  - Langfristig OIDC/JWT für Multi-Service/B2B-Integration.
- Nutzen: Zukunftssicheres Security-Modell.

### 3. Eventing / Async Beyond Celery
- Aktuell: Redis als Queue-Backbone.
- Verbesserung: Migration Richtung GCP Pub/Sub oder Cloud Tasks.
- Nutzen: Native Cloud-Integration, Monitoring & IAM.

### 4. Vector-DB Zukunftssicherheit
- Aktuell: pgvector im Cloud SQL.
- Verbesserung: Abstraktionsschicht (z. B. LangChain VectorStore Interface).
- Nutzen: Flexibilität für Weaviate, Milvus, Pinecone etc.

### 5. Observability Next-Level
- Aktuell: Langfuse für LLM, ELK für Logs, Cloud Monitoring für Infra.
- Verbesserung: OpenTelemetry als Dach für Logs, Metrics, Traces.
- Nutzen: Einheitliches Observability-Modell.

### 6. Frontend/UX
- Aktuell: Tailwind, Storybook.
- Verbesserung: 
  - Accessibility (WCAG/ARIA).
  - Internationalisierung (i18n).
- Nutzen: Modernes, barrierefreies, global nutzbares UI.

### 7. Testing / QA
- Aktuell: Unit, E2E, Smoke vorhanden.
- Verbesserung: 
  - Chaos-Tests (Fault Injection, Störfälle in Redis/Cloud SQL).
  - Lasttests (K6, Locust).
- Nutzen: Robustheit unter realen Bedingungen.

---

## Empfehlung
- **Quick Wins (nächste 1–2 Monate):**
  - API-Contract-First einführen.
  - Accessibility/i18n im Frontend.
  - Erste Chaos-/Load-Tests in CI.

- **Deep Changes (3–12 Monate):**
  - OIDC/mTLS Security-Architektur.
  - Migration zu Pub/Sub.
  - OpenTelemetry Rollout.
  - Vector-DB Abstraktion und Evaluierung alternativer Engines.

---

Mit diesen Maßnahmen bewegt sich NOESIS-2 von „Goldstandard“ zu einer **State-of-the-Art AI Plattform**.
