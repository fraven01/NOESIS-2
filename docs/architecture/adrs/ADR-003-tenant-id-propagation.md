# ADR-003: Explizite Tenant-Propagation

## Status
Accepted

## Kontext
`tenant_id` bestimmt Schema-Isolation. Aktuell wird sie im Web-Layer aus Headern gelesen und in Tasks/Tools weitergegeben, aber in einigen Pfaden implizit aus Repositories abgeleitet.

## Entscheidung
- `tenant_id` wird immer explizit als Feld/Parameter durchgereicht (HTTP → Celery → Graph → Tool). Keine Ableitung aus Dokument- oder Case-IDs.
- Worker- und Tool-Kontexte validieren `tenant_id` als Pflichtfeld (`extra=forbid` in Pydantic-Modellen).
- Tasks, die einen Router oder Vector-Client erzeugen, nehmen `tenant_schema` optional, aber nicht als Ersatz für `tenant_id`.

## Konsequenzen
- Multi-Tenancy bleibt streng: ein fehlendes oder leeres `tenant_id` führt zu sofortigem Fehler.
- Telemetrie (Logs/Spans) muss `tenant_id` als Attribut tragen; Sampling/Filter funktionieren mandantenspezifisch.
- Migrationen/Tests dürfen keine Default-Tenant annehmen; Fixtures müssen `tenant_id` setzen.

