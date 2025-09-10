# Tenant-Verwaltung

Dieses Dokument beschreibt, wie NOESIS 2 Mandanten (Tenants) realisiert und wie Administratoren neue Tenants anlegen und pflegen.

## Architektur

NOESIS 2 nutzt [django-tenants](https://django-tenants.readthedocs.io/) und speichert jeden Mandanten in einem eigenen PostgreSQL-Schema. Das Modell `Tenant` erweitert `TenantMixin` und erzeugt automatisch das Schema, während `Domain` den Hostnamen mit dem jeweiligen Mandanten verknüpft.

## Request-Routing

Der Middleware-Baustein `TenantSchemaMiddleware` liest den HTTP‑Header `X-Tenant-Schema` und speichert ihn in `request.tenant_schema`. Dekorator und Mixin in `common.tenants` prüfen diesen Wert und lehnen Anfragen ab, wenn er nicht zum aktiven Schema passt.

## Management-Kommandos

* `python manage.py bootstrap_public_tenant --domain <domain>` – legt das öffentliche Schema und die zugehörige Domain an.
* `python manage.py create_tenant --schema=<schema> --name=<name> --domain=<domain>` – erstellt einen neuen Mandanten und erzeugt automatisch das Schema.
* `python manage.py list_tenants` – listet alle vorhandenen Mandanten.
* `python manage.py create_demo_data` – legt einen Demo-Mandanten mit Beispielnutzer und Projekten an.

## Django-Admin

Im Django-Admin können `Tenant` und `Domain` verwaltet werden. Für ausgewählte Tenants steht die Aktion **Migrate selected tenants** zur Verfügung, die `migrate_schemas` aufruft, um Migrationen anzuwenden.

## Nutzung in Clients

Alle API-Aufrufe müssen den Header `X-Tenant-Schema` setzen, damit die Anfrage im richtigen Schema ausgeführt wird. Beispiel:

```
curl -H "X-Tenant-Schema: alpha" http://localhost:8000/tenant-demo/
```

Die Guards aus `common.tenants` können genutzt werden, um eigene Views gegen fehlerhafte Schemanutzung abzusichern.
