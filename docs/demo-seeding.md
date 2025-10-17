# Demo-Seeding Leitfaden

## Zweck
- Schnell reproduzierbare Dokument-Samples für das `demo`-Tenant-Schema.
- Kein persistentes Django-Modell mehr: Dokumente und Assets liegen im In-Memory-Repository des neuen `documents`-Subsystems.
- Befüllt wird das Repository über die Python-CLI (`python -m documents.cli`).

## Voraussetzungen
1. Tenant & Superuser anlegen (siehe [`docs/multi-tenancy.md`](multi-tenancy.md) und [`docs/tenant-management.md`](tenant-management.md)).
2. Backend-Container oder virtuelle Umgebung aktivieren, sodass `documents.cli` importiert werden kann.
3. Optional: Beispielquellen finden sich unter `documents/demo*.txt`.

## Schnelleinstieg

### Dokument hinzufügen
```bash
python -m documents.cli --json docs add \
  --tenant demo \
  --collection onboarding \
  --title "Mitbestimmungsleitfaden" \
  --inline-file documents/demo1.txt \
  --media-type text/plain \
  --source upload
```
- `--inline-file` liest eine Datei ein und konvertiert sie automatisch zu Base64.
- Alternativ kann `--inline` direkt mit einem Base64-String arbeiten oder `--file-uri` auf einen zuvor gespeicherten Blob verweisen.

### Dokumente auflisten
```bash
python -m documents.cli --json docs list --tenant demo --collection onboarding --limit 10
```
- Mit `--latest-only` wird pro Dokument nur die aktuellste Version zurückgegeben.
- Die Ausgabe enthält `next_cursor`, um große Mengen schrittweise zu laden.

### Dokument abrufen oder löschen
```bash
python -m documents.cli --json docs get --tenant demo --doc-id <UUID>
python -m documents.cli --json docs delete --tenant demo --doc-id <UUID>
```
- Mit `--hard` werden auch archivierte Versionen entfernt.

### Asset ergänzen
```bash
python -m documents.cli --json assets add \
  --tenant demo \
  --document <DOC_UUID> \
  --media-type image/png \
  --inline "$(base64 -w0 theme/static_src/logo.png)" \
  --caption-method manual \
  --text-description "Titelbild"
```
- `--inline-file` steht ebenfalls zur Verfügung.
- Auflistung: `python -m documents.cli --json assets list --tenant demo --document <DOC_UUID>`

## Skripting
- Wiederholbare Seeds lassen sich über Python-Skripte umsetzen (siehe Beispiele in [`docs/documents/cli-howto.md`](documents/cli-howto.md)).
- Für deterministische Inhalte empfiehlt sich eine feste UUID (`--doc-id`) und stabile `collection`-Werte.

## Aufräumen & Reset
- Das In-Memory-Repository wird beim Prozess-Neustart geleert.
- Manuelles Entfernen einzelner Einträge erfolgt über `docs delete` bzw. `assets delete`.

## Troubleshooting
- `storage_uri_missing`: Der referenzierte Blob wurde zuvor nicht gespeichert – `--inline`/`--inline-file` verwenden oder einen gültigen Pfad übergeben.
- `document_not_found`: UUID prüfen oder sicherstellen, dass der Prozess nicht neu gestartet wurde.
- `blob_source_required`: Bei `docs add` muss genau eine Quelle (`--inline`, `--inline-file` oder `--file-uri`) angegeben werden.

Weitere Beispiele und Logging-Hinweise finden sich in [`docs/documents/cli-howto.md`](documents/cli-howto.md) sowie in den Tests unter `tests/documents/test_document_cli.py`.
