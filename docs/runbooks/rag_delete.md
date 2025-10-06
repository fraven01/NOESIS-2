# Runbook: RAG-Dokumente löschen & pflegen

Kurzleitfaden für die Entfernung von RAG-Dokumenten in Produktions-tenants. Alle Schritte sind idempotent geplant – bei Unsicherheiten Vorgang abbrechen und erneut ausführen.

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
3. **Löschung durchführen:**
   ```sql
   DELETE FROM rag.documents
         WHERE tenant_id = :tenant
           AND document_id = ANY(:document_ids);
   ```
4. **Audit-Event schreiben:**
   ```bash
   noesis2-manage audit:log --event rag.hard_delete --tenant $TENANT --payload @payload.json
   ```
5. **Post-Steps:**
   ```bash
   psql -c "VACUUM (VERBOSE, ANALYZE) rag.documents"   # Storage zurückgewinnen
   psql -c "REINDEX TABLE CONCURRENTLY rag.documents"   # optional, bei großen Löschungen
   noesis2-manage router:invalidate --tenant $TENANT --scope rag
   ```

## Checkliste nach Abschluss
- [ ] Zählerabgleich: `SELECT COUNT(*)` vor/nach Soft- oder Hard-Delete dokumentiert.
- [ ] Stichproben-Query (`retrieve.hybrid`) liefert 0 Treffer für gelöschte IDs.
- [ ] Indizes im grünen Bereich (`pg_stat_user_indexes`) – keine reindex_pending-Werte.
- [ ] Ticket/Runbook-Eintrag mit Timestamp, Operator und Parametern aktualisiert.
