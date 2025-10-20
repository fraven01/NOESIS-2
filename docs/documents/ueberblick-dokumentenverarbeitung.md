# Dokumentenverarbeitung – Überblick

## Zweck
- Transformiert eingehende Normalized-Dokumente in angereicherte Artefakte mit Text-, Asset- und Chunk-Daten, damit nachgelagerte Retrieval- und Analyseprozesse konsistente Inputs erhalten.
- Gewährleistet Wiederaufnehmbarkeit durch zustandsbehaftete Verarbeitung und aktualisiert Metriken sowie Statistiken pro Workflowlauf.

## Hauptkomponenten
- Orchestrator steuert Parser, Captioning-Pipeline, Chunker sowie Repository- und Storage-Schnittstellen und überwacht den Fortschritt.
- Parser liefert strukturierte Textblöcke, extrahierte Assets und Statistikdaten für Folgephasen.
- Repository und Storage persistieren Dokumentversionen, Asset-Metadaten und Binärinhalte und erlauben idempotente Upserts.
- Captioning-Pipeline ergänzt visuelle Assets um Beschreibungen und Hashes, einschließlich OCR-Fallbacks.
- Chunker erzeugt Workflow-spezifische Ausgabeeinheiten und sammelt Kennzahlen für Downstream-Systeme.

## Verarbeitungsphasen
- **INGESTED** – Ausgangszustand mit initialen Metadaten vor der Analyse.
- **PARSED_TEXT** – Parserergebnisse sind gespeichert und Textblöcke stehen bereit.
- **ASSETS_EXTRACTED** – Binäre Assets sind persistiert und mit Dokumenten verknüpft.
- **CAPTIONED** – Bild-Assets wurden automatisch oder manuell beschriftet und mit Konfidenzwerten versehen.
- **CHUNKED** – Textabschnitte sind in Chunks segmentiert, Statistiken aktualisiert und bereit für Retrieval.

## Konfigurationsrahmen
- Sicherheits- und Robustheitsoptionen für Parser (Safe-Mode für PDF, Readability-HTML, Notes & Empty Slides).
- Steuerung der Asset-Anreicherung über OCR-Aktivierung, Captioning-Schalter sowie Fallback-Konfidenzen.
- Konfidenz-Schwellen pro Sammlung und Standardwerte zur Bewertung von Beschriftungen.
- Erweiterbarkeit durch injizierbare OCR-Renderer und austauschbare Chunker-Implementierungen.

## Zentrale Begriffe
- **ProcessingContext** bündelt unveränderliche Metadaten wie Tenant-, Workflow-, Dokument- und Collection-IDs.
- **ProcessingMetadata** normalisiert Eingangswerte, erzwingt Zeitzoneninformationen und referenziert Workflow-Läufe.
- **ParseArtifact** kapselt persistierte Textblöcke, Asset-Referenzen und Zwischenstatistiken für Wiederverwendung.
- **ChunkArtifact** enthält generierte Chunk-Payloads und aggregierte Kennzahlen.
- **ProcessingOutcome** liefert den finalen Dokumentzustand einschließlich optionaler Artefakte und aktualisiertem Kontext.
