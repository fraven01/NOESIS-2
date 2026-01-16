# Chunking-Audit für Crawler-Inhalte

## Zusammenfassung
- Die HTML-Normalisierung entfernt Scripts, Navigation und dekorative Elemente zuverlässig, liefert aber keine Gewichtung für wiederkehrende Boilerplate-Blöcke.
- Die Crawler-Ingestion nutzt zwar den Dokument-Graphen mit `SimpleDocumentChunker`, leitet für das Embedding jedoch das komplette normalisierte Dokument als Einmal-Chunk weiter.
- Die Produktions-Chunker in `ai_core.tasks.chunk` schneiden Texte entlang von Überschriften, wenden Token-Limits und Überlappung an und bauen Parent-Metadaten auf – diese Logik wird im Crawler-Pfad bislang nicht verwendet.
- Asset-Metadaten (Alt-Text, Kontext, Parent-Referenzen) werden am Parser erfasst, landen aber weder im finalen Chunk-Text noch im Chunk-Metadaten-Payload der Crawler-Pipeline.

## Harvesting & Parsing
- `_strip_boilerplate` entfernt Skripte, Style-Blöcke sowie Header/Footer/Nav-Container und fokussiert damit den Content-Root vor dem Block-Walk.【F:documents/parsers_html.py†L282-L312】
- `_HtmlState.add_text_block` normalisiert Whitespaces, respektiert Abschnitts-Hierarchien und hält Wort-, Absatz- und Listenzähler für spätere Statistiken bereit.【F:documents/parsers_html.py†L169-L205】
- `_HtmlState.add_asset` und `finalise_assets` verknüpfen Bilder mit Alt-/Caption-Texten, Kontextfenstern und Eltern-Referenzen (z. B. `section:…`) und ergänzen diese Metadaten um Locator-IDs.【F:documents/parsers_html.py†L207-L278】
- Während `_ensure_normalized_payload` die Parsergebnisse prefetched, ersetzt es die Primärtexte durch den strukturierten Block-Join, sodass Boilerplate aus dem Normalisierungs-Ergebnis verbannt wird.【F:ai_core/graphs/crawler_ingestion_graph.py†L514-L563】

## Chunking-Befund
- Der Crawler-Graph instanziiert standardmäßig `SimpleDocumentChunker`, der lediglich einen Chunk pro Textblock (max. 2048 Byte) generiert – gedacht als Smoke-Test-Preview, nicht für produktive Embeddings.【F:ai_core/graphs/crawler_ingestion_graph.py†L185-L204】【F:documents/cli.py†L504-L562】
- Beim eigentlichen `trigger_embedding` wird dennoch **das gesamte normalisierte Dokument** als einzelner Chunk (`Chunk(content=normalized_content, …)`) an den Vector-Client gesendet. Parent-IDs, Abschnittsgrenzen oder Overlaps fehlen vollständig.【F:ai_core/api.py†L413-L478】
- Das robuste Chunking mit Token-Zielgrößen (Default 450), Hard-Limit, dynamischer Überlappung, Abschnittsregistrierung und Parent-Capture steht in `ai_core.tasks.chunk` bereit, wird aber nur in der generischen Upload/Batch-Ingestion genutzt.【F:ai_core/tasks/ingestion_tasks.py†L700-L1257】
- Damit laufen selbst lange Crawler-Dokumente in einen einzigen Vektor – eine häufige Ursache für `vector_candidates: 0`, weil Suchbegriffe semantisch verwässern und keine feinkörnigen Treffer existieren.

## Boilerplate & Irrelevanzen
- HTML-spezifische Boilerplate wird aggressiv gestrippt; für textuelle Boilerplate (z. B. wiederholte Job-Listings) fehlt hingegen ein Score oder eine Quarantäne-Strategie. In der Chunk-Phase gibt es keinen weiteren Filter.
- Ohne produktiven Chunker landen Listen oder FAQ-Sektionen komplett im Volltext-Chunk; in der Upload-Pipeline würden sie dagegen in eigene Token-Begrenzte Chunks fallen.【F:ai_core/tasks/ingestion_tasks.py†L940-L1159】

## Asset-Verknüpfung
- Parser liefern strukturierte Asset-Metadaten inkl. `parent_ref`, Caption-Kandidaten und Kontextfenster.【F:documents/parsers_html.py†L207-L278】
- `SimpleDocumentChunker` kopiert nur `parent_ref` in die Preview-Chunks.【F:documents/cli.py†L533-L560】
- `trigger_embedding` übergibt keine Asset-Informationen; das finale Chunk-Metadaten-Objekt enthält lediglich Tenant/Source/Hashes und verliert jede Asset-Lokalisierung.【F:ai_core/api.py†L432-L468】

## Empfehlungen
1. **Produktions-Chunker verwenden**  
   - Integriert `ai_core.tasks.chunk` (inkl. Parent/Overlap-Logik) in den Crawler-Graphen, bevor `trigger_embedding` aufgerufen wird. Alternativ: `trigger_embedding` soll `chunks.json` konsumieren statt `normalized_content` zu versenden.【F:ai_core/tasks/ingestion_tasks.py†L700-L1257】【F:ai_core/api.py†L413-L478】
2. **Chunk-Metadaten anreichern**  
   - Übernehmt `parent_ids`, Abschnittstitel und ggf. Asset-Locator in das Meta-Payload, damit Retrieval gezielt Eltern-Kontext (z. B. Überschriften) filtern kann.【F:ai_core/tasks/ingestion_tasks.py†L1112-L1188】
3. **Boilerplate-Heuristiken erweitern**  
   - Ergänzt `_strip_boilerplate` um Freitext-Signaturen (z. B. reguläre Ausdrücke für Cookie-Banner/Job-Listings) oder bewertet Blöcke anhand Wiederholungs-Score, bevor sie als Chunk-Kandidaten dienen.【F:documents/parsers_html.py†L282-L312】
4. **Asset-Kontext fusionieren**  
   - Fügt Caption/Alt-Kontext beim Chunk-Bau dem unmittelbaren Text-Chunk hinzu (z. B. Prefix „Figure: …“) oder referenziert Assets über zusätzliche Chunk-Metafelder, damit Embeddings Bildbeschreibungen berücksichtigen.【F:documents/parsers_html.py†L207-L278】【F:ai_core/api.py†L432-L468】
5. **Logging & Observability ausbauen**  
   - Loggt Chunk-Anzahl, Token-Statistiken und Overlap pro Dokument (ähnlich wie Upload-Pipeline) sowie Warnungen, wenn nur ein Chunk erzeugt wurde. Ergänzt `EmbeddingResult` um `chunk_count` und Parent-Capture-Indikatoren für schnellere Diagnosen.【F:ai_core/api.py†L413-L478】

Mit diesen Anpassungen werden Crawler-Dokumente in kohärente, überschaubare Chunks zerlegt, Assets bleiben auffindbar und Retrieval-Anfragen wie „Cellebrite“ landen künftig in konkreten Vektoren statt in einem unhandlich großen Einheitschunk.
