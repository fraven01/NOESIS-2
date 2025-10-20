# Persistenz & Public API

## Persistenz-Prinzipien
- **Idempotenz:** Wiederholte Pipeline-Läufe erkennen bestehende Dokumente und Assets per Workflow- und Asset-ID, überspringen unnötige Re-Verarbeitungen und behalten bestehende Speicherzeiger bei.【F:documents/tests/test_pipeline.py†L467-L513】【F:documents/tests/test_pipeline.py†L603-L635】
- **Unveränderliche Blobs:** Inline-Daten werden vor dem Speichern in Datei-Blobs umgewandelt; Checksummen-Abweichungen lösen Fehler aus und verhindern nachträgliche Modifikationen.【F:documents/repository.py†L727-L763】【F:documents/contracts.py†L260-L281】
- **Content-Hashing:** Jeder Persistenzschritt hinterlegt SHA-256-Checksummen für Dokument- und Asset-Blobs, erzwingt Hex-Validierung und kontrolliert optional strikte Gleichheit.【F:documents/contracts.py†L150-L199】【F:documents/contracts.py†L760-L819】
- **Provenienz:** Metadaten bewahren Ursprung, Crawl-Timestamp, externe Referenzen und Parser-Statistiken; Assets tragen Kontext, Quelle und Caption-Herkunft zur Rückverfolgbarkeit.【F:documents/contracts.py†L331-L378】【F:documents/contracts.py†L539-L610】
- **Versionierung:** Dokument-Referenzen kapseln Workflow, Collection und optionale Versionsmarker; Repository-Listings liefern geordnete Cursors für neueste Revisionen.【F:documents/contracts.py†L150-L199】【F:documents/repository.py†L270-L312】【F:documents/repository.py†L784-L816】

## DocumentsRepository (Quick-Ref)
- **Zweck:** Zentrale Persistenzschicht für normalisierte Dokumente und Assets mit Workflow-Scopes, Cursor-Listings und Soft-/Hard-Deletes.【F:documents/repository.py†L180-L269】【F:documents/repository.py†L270-L345】
- **Zusagen:** Upserts replizieren vollständige Modelle, materialisieren Blobs, synchronisieren Asset-Indizes und liefern stets Kopien; List-Operationen sortieren deterministisch nach Zeitstempel, Dokument-ID und Version.【F:documents/repository.py†L182-L268】【F:documents/repository.py†L314-L345】【F:documents/repository.py†L784-L803】
- **Fehlerszenarien:** Workflow- oder Asset-Mismatches, ungültige Cursor, Checksum-Kollisionen oder fehlende Storage-Einträge schlagen mit ValueError/KeyError fehl und blockieren Persistenzfehler früh.【F:documents/repository.py†L186-L205】【F:documents/repository.py†L744-L782】【F:documents/storage.py†L49-L57】

## Storage (Quick-Ref)
- **Zweck:** Minimale Binärspeicher-Schnittstelle, die URIs mit abgeleiteten Checksummen und Größen zurückgibt und Retrieval ermöglicht.【F:documents/storage.py†L12-L47】
- **Zusagen:** `put` erzeugt deterministische SHA-256-Werte und `memory://`-URIs pro Payload, `get` liefert Byte-identische Daten und protokolliert Zugriff.【F:documents/storage.py†L34-L62】
- **Fehlerszenarien:** Nicht unterstützte URI-Schemata lösen `storage_uri_unsupported` aus, fehlende Einträge melden `storage_uri_missing`.【F:documents/storage.py†L49-L57】

## Parser-Dispatcher (Quick-Ref)
- **Zweck:** Hält eine geordnete Parser-Registry und ruft den ersten Parser auf, der ein Dokument verarbeiten kann.【F:documents/parsers.py†L309-L343】
- **Zusagen:** Registrierungen validieren `can_handle`/`parse`, erhalten die Einfügereihenfolge und leiten `parse`-Aufrufe unverändert weiter.【F:documents/parsers.py†L312-L352】
- **Fehlerszenarien:** Fehlende Fähigkeiten oder keine passenden Parser führen zu `TypeError` bzw. `no_parser_found` und signalisieren Konfigurationslücken.【F:documents/parsers.py†L318-L343】

## Pipeline-Konfiguration (Quick-Ref)
- **Zweck:** Bündelt Sicherheits- und Anreicherungs-Flags, Caption-Confidence-Grenzen sowie optionale Renderer für die Verarbeitungspipeline.【F:documents/pipeline.py†L127-L142】
- **Zusagen:** Normalisiert Boolesche Flags, Konfidenzgrenzen und Collection-Mappings, verweigert ungültige Renderer und berechnet effektive Thresholds pro Sammlung.【F:documents/pipeline.py†L144-L220】
- **Fehlerszenarien:** Falsche Typen, außer Range liegende Thresholds oder nicht auflösbare Collection-Keys werfen ValueErrors und schützen vor fehlerhafter Konfiguration.【F:documents/pipeline.py†L100-L201】

## Processing-Kontext (Quick-Ref)
- **Zweck:** Trägt unveränderliche Metadaten über Tenant, Workflow, Dokument, Quelle und Zeitstempel durch alle Zustandsübergänge.【F:documents/pipeline.py†L240-L309】
- **Zusagen:** Kontextinstanzen bleiben unveränderlich, `transition` erzeugt neue Kontexte mit validierten Zuständen, und `from_document` rekonstruiert Metadaten inklusive Versions- und Workflow-Bindung.【F:documents/pipeline.py†L311-L359】
- **Fehlerszenarien:** Fehlende Workflow-Angaben, naive Zeitstempel oder unbekannte Statuswerte werden als ValueError abgewiesen und verhindern inkonsistente Laufzeitkontexte.【F:documents/pipeline.py†L249-L275】【F:documents/pipeline.py†L293-L359】
