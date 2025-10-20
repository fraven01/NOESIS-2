# Agentenauftrag

## Mission & Scope
- Dokumentenverarbeitung orchestrieren: Parser, Chunking, Kontextanreicherung.
- Asset-Pipeline steuern: Extraktion, Captioning, Hash-basierte Persistenz.
- Persistenzschichten konsistent halten: Repository, Storage, Metadatenlaufwerke.

## Design-Prinzipien
- Deterministische Abläufe, Idempotenz pro Schritt, keine versteckten Seiteneffekte.
- Guards vor riskanten Operationen: Dateigröße, Strukturvalidierung, Timeout-Pfade.
- Normalisierung erzwingen: Zeitstempel, Lokalisierung, Encoding, eindeutige IDs.

## Öffentliche Oberflächen
- ProcessingContext & Metadata-Contracts: readonly, vollständig serialisierbar.
- Parser Dispatcher & Asset Pipeline: pure Inputs→Artefakte mit expliziten Stati.
- Repository/Storage APIs: Hash-Adressen, versionierte Blobs, unveränderliche Rückgaben.
- Konfiguration: Feature-Flags, Grenzwerte, Confidence-Schwellen via zentraler Registry.

## Qualitätsregeln
- Unit- und Regressionstests pflegen, Grenzfälle gemäß dokumentierten Limits abdecken.
- Ressourcenlimits einhalten: Speicher, Seitencount, Timeout-Margen, Batchgrößen.
- Serialisierbarkeit sicherstellen für Kontexte, Artefakte, Event-Payloads.
- Telemetrie & Auditing: Metriken, Trace-IDs, Provenienzfelder immer füllen.

## Erweiterungspunkte
- Neue Parser registrieren über Dispatcher-Vertrag, Artefakt-Schema beibehalten.
- Neue Asset-Adapter: deterministische Hashes, optionale Caption-Provider respektieren.
- Konfigurierbare Guards ergänzen: Schwellen, Whitelists, Feature-Toggles.
- Persistenz-Erweiterungen nur via versionierte Contracts und reversible Migrationen.
