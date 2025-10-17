# Tenant-Verwaltung

Dieses Dokument beschreibt, wie NOESIS 2 Mandanten (Tenants) realisiert und wie Administratoren neue Tenants anlegen und pflegen.

## Lokales Setup (nach Pull)

1. Abhängigkeiten aktualisieren
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`
2. Gültige PostgreSQL-Zugangsdaten in `.env` hinterlegen. Das Projekt nutzt `django_tenants.postgresql_backend`.
3. Nach jedem Pull alle Schemata migrieren:
   ```bash
   python manage.py migrate_schemas
   ```
4. Öffentliches Schema initialisieren:
   ```bash
   python manage.py bootstrap_public_tenant --domain <domain>
   ```
5. Mandanten anlegen:
   ```bash
   python manage.py create_tenant --schema=<schema> --name=<name> --domain=<domain>
   ```
6. Optional: vorhandene Tenants auflisten (`list_tenants`). Dokument-Datasets werden über `documents.cli` gepflegt (siehe `docs/documents/cli-howto.md`).
7. Eigene Views mit `tenant_schema_required` oder `TenantSchemaRequiredMixin` schützen, damit nur Anfragen mit korrektem Header akzeptiert werden.

## Architektur

NOESIS 2 nutzt [django-tenants](https://django-tenants.readthedocs.io/) und speichert jeden Mandanten in einem eigenen PostgreSQL-Schema. Das Modell `Tenant` erweitert `TenantMixin` und erzeugt automatisch das Schema, während `Domain` den Hostnamen mit dem jeweiligen Mandanten verknüpft.

## Request-Routing

Die Auswahl des Mandanten (Tenant) erfolgt primär über den Hostnamen der Anfrage, wie von `django-tenants` vorgesehen. Ein optionaler Schutzmechanismus in `common.tenants` (`tenant_schema_required` Decorator und `TenantSchemaRequiredMixin`) stellt sicher, dass der `X-Tenant-Schema`-Header explizit gesetzt wird und mit dem aktiven Schema übereinstimmt. Dies dient als zusätzliche Sicherheitsmaßnahme (Defense-in-Depth), um fehlgeleitete Anfragen zu verhindern. In Entwicklungs- und Testumgebungen (`DEBUG=True`) erlaubt die `HeaderTenantRoutingMiddleware` zusätzlich, den Mandanten direkt über den `X-Tenant-Schema`-Header zu wechseln, um lokale Tests ohne Domain-Konfiguration zu vereinfachen.

## Management-Kommandos

* `python manage.py bootstrap_public_tenant --domain <domain>` – legt das öffentliche Schema und die zugehörige Domain an.
* `python manage.py create_tenant --schema=<schema> --name=<name> --domain=<domain>` – erstellt einen neuen Mandanten und erzeugt automatisch das Schema.
* `python manage.py list_tenants` – listet alle vorhandenen Mandanten.
* Dokumentbeispiele: siehe `python -m documents.cli --help` oder die Beispiele in `docs/documents/cli-howto.md`.

## Django-Admin

Im Django-Admin können `Tenant` und `Domain` verwaltet werden. Für ausgewählte Tenants steht die Aktion **Migrate selected tenants** zur Verfügung, die `migrate_schemas` aufruft, um Migrationen anzuwenden.

## Nutzung in Clients

Für API-Endpunkte, die durch `tenant_schema_required` geschützt sind, muss der `X-Tenant-Schema`-Header gesetzt werden und mit dem durch den Hostnamen aufgelösten Schema übereinstimmen. Beispiel:

```
curl -H "X-Tenant-Schema: alpha" http://localhost:8000/tenant-demo/
```

Die Guards aus `common.tenants` können genutzt werden, um eigene Views gegen fehlerhafte Schemanutzung abzusichern.

## Testen

- Automatisiert: `pytest -q` führt die vorhandenen Tenant-Tests aus.
- Manuell:
  - Erfolgreiche Anfrage mit gesetztem Header:
    ```bash
    curl -H "X-Tenant-Schema: alpha" http://localhost:8000/tenant-demo/
    ```
  - Ohne oder falschen Header sollte `403 Forbidden` zurückgeben.
