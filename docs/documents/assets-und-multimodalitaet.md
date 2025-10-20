# Assets & Multimodalität

## Herkunft & Extraktion
- Parser liefern pro gefundenem Bild, Diagramm oder externen Verweis einen `ParsedAsset`; die Pipeline berechnet deterministische Asset-IDs aus Dokument und Locator, prüft bestehende Checksummen und legt nur neue Blobs in Storage ab.
- Bei eingebetteten Binärdaten werden SHA-256-Hashes und optionale Perzeptions-Hashes erzeugt; Dateien aus Referenzen werden nachgeladen oder als externe Pointer mit Ursprungshinweis registriert.

## Asset-Felder & Kontext
- Jedes Asset trägt Referenzen auf Tenant, Workflow, Dokument und optional Collection sowie Medien-Typ, Bounding Box, Seitenindex und optionale Eltern-Referenz für die Verortung.
- Kontext vor und nach dem Asset wird gekürzt gespeichert, OCR-Ergebnisse bleiben separat erhalten, und `text_description` ist auf kurze Beschreibungen begrenzt, ergänzt durch Ursprung (`origin_uri`), Klassifikation (`asset_kind`) und Integritätswerte (`checksum`, `created_at`).

## Multimodale Beschreibungen
- Bereits vorhandene Beschreibungen aus Parser-Metadaten (Alt-Text, Caption-Kandidaten, Notizen, Quellen) werden priorisiert und mit festen Confidence-Werten je Quelle übernommen.
- Fehlt eine Beschreibung, baut die Caption-Pipeline aus Umfeldtext einen Kontextstring (max. 512 Byte) und ruft das Multimodalmodell; akzeptierte Ergebnisse speichern Modellnamen, Methode `vlm_caption` und gemeldete Confidence.
- Bleibt das Modell unter dem Schwellenwert oder liefert kein Ergebnis, greift ein OCR-Fallback mit definierter Confidence; ansonsten bleibt der ursprüngliche Zustand erhalten.

## Confidence-Steuerung
- Mindest-Confidence pro Collection stammt aus der Pipeline-Konfiguration (`caption_min_confidence_default` plus overrides) und kann durch Policies pro Tenant, Collection oder Workflow weiter angehoben werden.
- Der OCR-Fallback nutzt eine separate Konstante (`ocr_fallback_confidence`), während manuelle Quellen feste Confidence-Mappings verwenden; ungültige Werte führen zu Warnungen oder Validierungsfehlern.

## Lebenszyklus
- **Extraktion:** Parser-Artefakte werden persistiert, Storage-Blobs geschrieben oder referenziert und Asset-Metadaten mit Checksummen und Kontext versehen.
- **Optionale Caption:** Die Caption-Pipeline aktualisiert nur bildfähige Assets ohne gültige Beschreibung und protokolliert Treffer, OCR-Fälle und Überspringen.
- **Persistenz:** Aktualisierte Dokumente und Assets werden im Repository gespeichert; Re-Runs erkennen unveränderte Checksummen und vermeiden doppelte Speichervorgänge.
