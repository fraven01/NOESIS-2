# Multi‑Tenancy Leitfaden

Dieser Leitfaden beschreibt Einrichtung, Betrieb und Besonderheiten der Mandantenfähigkeit (django-tenants) in NOESIS 2.

## Architektur
- Public-Schema: enthält ausschließlich `customers` (Tenants/Domains).
- Tenant-Schemata: enthalten alle übrigen Apps inkl. `auth/admin` und dem Custom-User `users.User`.
- Domain → Schema-Auswahl: `django_tenants` wählt anhand von `customers_domain` das Schema und setzt `connection.schema_name`.
- Optionaler Schutz für API/Views: `X-Tenant-Schema`-Header muss zum aktiven Schema passen (Decorator/Mixin aus `common.tenants`).

## Lokales Setup nach Pull
1) Abhängigkeiten aktualisieren
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`
2) PostgreSQL-Zugangsdaten in `.env` setzen. Engine: `django_tenants.postgresql_backend`.
3) Migrationen anwenden (alle Schemata):
   - `python manage.py migrate_schemas`
4) Public-Schema initialisieren:
   - `python manage.py bootstrap_public_tenant --domain localhost`
5) Demo-Tenant + Superuser (Beispiel):
   - `python manage.py create_tenant --schema=demo --name="Demo Tenant" --domain=demo.localhost`
   - `python manage.py create_tenant_superuser --schema=demo --username=demo --password=<PASSWORT>`
6) Hosts-Datei ergänzen (Windows als Admin: `C:\Windows\System32\drivers\etc\hosts`):
   - `127.0.0.1 demo.localhost`
   - optional weitere Tenants: `127.0.0.1 acme.localhost`
7) Weitere Tenants:
   - `python manage.py create_tenant --schema=acme --name="ACME Inc." --domain=acme.localhost`
   - `python manage.py create_tenant_superuser --schema=acme --username=admin --password=<PW>`
   - `python manage.py list_tenants`

Tipp: Alternativ zu Hosts-Einträgen kann `lvh.me` genutzt werden (Wildcard → 127.0.0.1), z. B. `acme.lvh.me`.

## Admin & Operator‑Rollen
- Per‑Tenant Admin (Standard):
  - Zugriff pro Domain, z. B. `http://demo.localhost:8000/admin/`.
  - Benutzerverwaltung und Daten nur innerhalb des aktiven Tenants.
  - Demo-Login: über `create_tenant_superuser` frei wählbar (Standardempfehlung: `demo`/`demo` im lokalen Setup).
  - Operator (Public):
  - Verwaltung von Tenants/Domains erfolgt per CLI‑Commands, kein Public‑Admin‑Login (da User per Tenant).
  - Wichtige Befehle: `bootstrap_public_tenant`, `create_tenant`, `list_tenants`, `create_tenant_superuser`.
  - Kunden-Admin (customers) ist nicht im Tenant‑Admin sichtbar.

### Staging ohne eigene Domain (Cloud Run)
- Cloud Run stellt einen Default‑Host bereit, z. B. `https://<service>-<hash>.<region>.run.app/`.
- Ohne eigene Domain kann nur genau ein Tenant über diese Hostname aufgelöst werden.
- Vorgehen für Staging:
  1) Public‑Domain auf den Cloud‑Run‑Host setzen:
     - `python manage.py bootstrap_public_tenant --domain <run.app-host>`
  2) Demo‑Tenant/Superuser anlegen:
     - `python manage.py create_tenant --schema=demo --name="Demo Tenant" --domain=<run.app-host>`
     - `python manage.py create_tenant_superuser --schema=demo --username=demo --password=<PASSWORT>`
  3) Cloud‑Run‑Host dem Demo‑Tenant zuordnen (zusätzliche Domain):
     - `python manage.py add_domain --schema=demo --domain=<run.app-host> --primary`
- Multi‑Tenant per Subdomain ist ohne eigene Domain/Wildcard nicht möglich. Für mehrere Tenants: separate Services oder später eigene Domain mit Wildcard DNS einrichten.

## X‑Tenant‑Schema Header
- Warum: Defense‑in‑Depth – Clients müssen das Ziel‑Schema explizit bestätigen.
- Wie: Middleware liest `X-Tenant-Schema` und Guards gleichen mit `connection.schema_name` ab.
- Testen (cURL):
  - `curl -H "Host: demo.localhost" -H "X-Tenant-Schema: demo" http://127.0.0.1:8000/tenant-demo/`
- Browser: Setzt keine Custom‑Header. Für geschützte Views Postman/cURL verwenden oder Guards auf reinen HTML‑Seiten im Dev deaktivieren.

## Troubleshooting
- 403 bei geschützten Views: `X-Tenant-Schema` fehlt/falsch.
- 404 bei `http://127.0.0.1:8000/`: Domain nicht hinterlegt. Entweder `localhost` nutzen oder `bootstrap_public_tenant --domain 127.0.0.1` ausführen.
- Migration-Fehler "Spalte existiert nicht": `python manage.py migrate_schemas` ausführen; bei neuen nicht‑null Feldern Drei‑Phasen‑Migration verwenden (siehe unten).
- Admin kann Users nicht finden: Admin/Users existieren pro Tenant; zuerst `create_tenant_superuser` für das Ziel‑Schema ausführen.

### Drei‑Phasen‑Migration (kurzer Verweis)
Für neue nicht‑null Felder nutze das Drei‑Phasen‑Muster (AddField → Backfill → AlterField). Die ausführliche Beschreibung inklusive Beispiel findest du im Entwickler‑Leitfaden: AGENTS.md, Abschnitt „Datenmigrationen (PostgreSQL Best Practices)“.
