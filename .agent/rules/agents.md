---
trigger: always_on
---

# NOESIS 2 Context & Governance Rule

## 1. Das Verfassungs-Prinzip (Root Law)

Jede Änderung und jeder Plan **MUSS** zuerst die Wurzel-Datei `AGENTS.md` konsultieren.

- Sie definiert die unveränderliche **4-Layer-Architektur**, **ID-Semantik** (`tenant_id`, `trace_id`) und **Tool-Verträge**.
- Verstöße gegen `AGENTS.md` sind untersagt, es sei denn, eine lokale Regel erlaubt dies explizit oder es ist sinnoll, dann nur beschreiben, dass man dagegen verstößt und warum. (Ausnahme-Genehmigung).

## 2. Das Lokalitäts-Prinzip (Local Law)

Bevor Code in einem Unterverzeichnis (z. B. eine Django App oder ein Modul) erstellt oder geändert wird, **MUSS** der Agent nach lokalen Kontext-Dateien suchen:

- **Priorität 1:** `AGENTS.md` im aktuellen oder übergeordneten Verzeichnis (z. B. `theme/AGENTS.md`).
- **Priorität 2:** `README.md` im App-Root (z. B. `ai_core/rag/README.md`, `ai_core/llm/README.md`).

## 3. Konfliktlösung

- **Spezifisch schlägt Generisch:** Anweisungen in einer app-spezifischen `README.md` (z. B. Chunking-Strategien in `ai_core/rag/README.md`) überschreiben generische Best-Practices, solange sie die System-Integrität (Tenant-Isolation) nicht gefährden.
- **Referenz-Pflicht:** Wenn eine lokale `README.md` existiert, muss die Lösung zitieren, welche Regel daraus angewendet wurde.

## 4. Pflicht-Check vor Implementierung

Führe folgenden mentalen Check aus:

1. [ ] Habe ich `AGENTS.md` für globale Layer-Regeln gelesen?
2. [ ] Habe ich die `README.md` der Ziel-App (z. B. `ai_core/graphs/`) auf spezifische Design-Patterns geprüft?
3. [ ] Verletze ich keine `tenant_id` Propagations-Regeln?
