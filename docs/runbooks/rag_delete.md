# Runbook: RAG-Dokumente löschen & pflegen

Kurzleitfaden für die Entfernung von RAG-Dokumenten in Produktions-tenants. Alle Schritte sind idempotent geplant – bei Unsicherheiten Vorgang abbrechen und erneut ausführen.

> **Neu:** Die zuvor manuelle Hard-Delete-Prozedur ist jetzt als Celery-Task `rag.hard_delete` verfügbar. Der Task kapselt SQL-Löschung, Audit-Log und Cache-Vacuumierung. Die untenstehende SQL-Variante bleibt als Fallback dokumentiert.
>
> **Worker-Hinweis:** Die Queue `rag_delete` wird von den Standard-Workern verarbeitet (`celery -A noesis2 worker -l info -Q celery,rag_delete`). Prüfe vor einem Lauf, dass der Worker aktiv ist.

## E1 Soft-Delete (Standardweg)
1. **Dokumente markieren:**
   ```sql
   UPDATE rag.documents
      SET deleted_at = NOW()
    WHERE tenant_id = :tenant
      AND document_id = ANY(:document_ids)
      AND deleted_at IS NULL;
   ```
2. **Idempotenz prüfen:** Folgeausführung mit denselben Parametern sollte `UPDATE 0` zurückgeben.
3. **Router-Cache spülen:**
   ```bash
   noesis2-manage router:invalidate --tenant $TENANT --scope rag
   ```
4. **Smoke-Test:** Retrieve-Call gegen denselben Tenant liefert keine Treffer mehr.

## E2 Hard-Delete (Ausnahmefall)
1. **Freigabe & Scope:** Admin bestätigt Scope (`tenant_id`, `project_id`) und referenziert Support-Ticket.
2. **Bestätigung einholen:** Zwei-Faktor-Approval dokumentieren (z. B. Slack-Thread + Jira-Comment).
3. **Endpoint auslösen (präferierter Weg):**
   ```bash
   curl -X POST "https://api.noesis.example/ai/rag/admin/hard-delete/" \
     -H "Content-Type: application/json" \
     -H "X-Internal-Key: <SERVICE-KEY>" \
     -d '{
       "tenant_id": "<TENANT-UUID>",
       "document_ids": ["<DOC-ID-1>", "<DOC-ID-2>"],
       "reason": "cleanup",
       "ticket_ref": "TCK-1234"
     }'
   ```
   - Admin-Sessions können den Endpoint ohne `X-Internal-Key` nutzen; Service-zu-Service-Flows setzen den Header mit einem Key aus `RAG_INTERNAL_KEYS`.
   - Die Response enthält `trace_id` und `job_id` für das Audit. `X-Trace-Id` wird zusätzlich als Header zurückgegeben.
4. **Direkter Task-Aufruf (Alternative):**
   ```python
   from ai_core.rag.hard_delete import hard_delete

   hard_delete.delay(
       tenant_id,
       document_ids,
       reason="cleanup",
       ticket_ref="TCK-1234",
       actor={"internal_key": "<SERVICE-KEY>"},
   )
   ```
   - Alternativ kann ein aktiver Admin-User über `actor={"user_id": <pk>}` autorisieren.
   - Die Task-Response enthält `documents_deleted`, `not_found`, `deleted_ids`, sowie `vacuum_performed`/`reindex_performed` und erzeugt automatisch ein Audit-Log (`rag.hard_delete.audit`).
5. **Fallback (nur falls Task nicht verfügbar ist):** Manuelle SQL-Löschung durchführen und anschließend Audit-/Maintenance-Schritte ausführen:
   ```sql
   DELETE FROM rag.documents
         WHERE tenant_id = :tenant
           AND document_id = ANY(:document_ids);
   ```
   ```bash
   noesis2-manage audit:log --event rag.hard_delete --tenant $TENANT --payload @payload.json
   psql -c "VACUUM (VERBOSE, ANALYZE) rag.documents"   # Storage zurückgewinnen
   psql -c "REINDEX TABLE CONCURRENTLY rag.documents"   # optional, bei großen Löschungen
   noesis2-manage router:invalidate --tenant $TENANT --scope rag
   ```

## Checkliste nach Abschluss
- [ ] Zählerabgleich: `SELECT COUNT(*)` vor/nach Soft- oder Hard-Delete dokumentiert.
- [ ] Stichproben-Query (`retrieve.hybrid`) liefert 0 Treffer für gelöschte IDs.
- [ ] Indizes im grünen Bereich (`pg_stat_user_indexes`) – keine reindex_pending-Werte.
- [ ] Ticket/Runbook-Eintrag mit Timestamp, Operator und Parametern aktualisiert.
