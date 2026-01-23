======================================================================
DOCUMENT CHUNK REVIEW REPORT
======================================================================

Document ID:  0e52f8e9-3054-4171-9af3-ca0f7dd58728
Tenant ID:    a07960c9-32bc-5dba-a0bc-02a1b4fe1f2f
Schema:       rag
Generated:    2026-01-23T17:32:09.613525
Total Chunks: 3

----------------------------------------------------------------------
CHUNK STATISTICS
----------------------------------------------------------------------
  Total characters: 863
  Min length:       112 chars
  Max length:       380 chars
  Mean length:      288 chars
  Median length:    371 chars
  Std deviation:    152 chars

  Size distribution:
    tiny (<100)          0 
    small (100-300)      1 █
    medium (300-800)     2 ██
    large (800-1500)     0 
    huge (>1500)         0 

----------------------------------------------------------------------
COVERAGE
----------------------------------------------------------------------
  Coverage unavailable (no normalized document text found).

----------------------------------------------------------------------
CHUNK BOUNDARY ANALYSIS
----------------------------------------------------------------------
  Found 1 potentially problematic boundaries:

  --- After Chunk 0 / Before Chunk 1 ---
  Issues: ends_mid_sentence
  End:   "...| Chunk 1 von 3
Anlage 1 KBV Rahmen IT 2.0 Systembeschreibung Systembeschreibung"
  Start: "Anlage 1 Systembeschreibung.pdf | Chunk 2 von 3
Allgemeine Informationen: Wo wir..."


----------------------------------------------------------------------
DETECTED PROBLEMS
----------------------------------------------------------------------
  Chunk 0:
    ℹ️ [incomplete_sentence] Chunk doesn't end with sentence punctuation

----------------------------------------------------------------------
LLM QUALITY EVALUATION
----------------------------------------------------------------------
  Mean Coherence:           68.3/100
  Mean Completeness:        58.3/100
  Mean Reference Resolution:63.3/100
  Mean Redundancy:          79.0/100
  Mean Overall:             67.2/100

  Lowest scoring chunks:
    - Chunk 096c9e3f...: 38/100
      Reason: The chunk is metadata-like (file name, chunk index, title) and lacks a self-contained semantic unit.
    - Chunk 044f06fd...: 77/100
      Reason: The chunk presents a coherent set of questions about provider-side system information and data flows
    - Chunk d61bfde0...: 88/100
      Reason: The chunk presents a coherent set of questions about the system's usage and deployment. It is self-c

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
  1. Low average coherence (68.3). Consider semantic chunking strategy.
  2. Low average completeness (58.3). Chunks may need more context. Increase overlap or chunk size.

----------------------------------------------------------------------
CHUNK CONTENT
----------------------------------------------------------------------
Chunk 0 (ID: 096c9e3f-ee25-522e-a6d7-a1ae69158199) - Length: 112
--------------------
Anlage 1 Systembeschreibung.pdf | Chunk 1 von 3
Anlage 1 KBV Rahmen IT 2.0 Systembeschreibung Systembeschreibung

======================================================================
Chunk 1 (ID: d61bfde0-1239-5c72-ba23-fca7ecd243b9) - Length: 380
--------------------
Anlage 1 Systembeschreibung.pdf | Chunk 2 von 3
Allgemeine Informationen: Wo wird das System eingesetzt (Unternehmen)? In welchen Fachbereichen soll das System gesetzt werden? Wenn mehr als eine Software eingeführt wird: Welche Softwareanwendungen sind umfasst? (Hersteller und Produkte müssen benannt werden) 4. Cloudsoftware – SaaS, PaaS, IaaS? Wofür wird das System eingesetzt?

======================================================================
Chunk 2 (ID: 044f06fd-c93f-53f3-9788-ed273973eab9) - Length: 371
--------------------
Anlage 1 Systembeschreibung.pdf | Chunk 3 von 3
Anbieterseitige Informationen zum System internes Systemhandbuch (aktuelle Version), soweit vorhanden Welche Systeme bekommen Daten von diesem System? Wenn Altsysteme ersetzt werden: welche Systeme sind das? Wenn einzelne Funktionen ersetzt werden: welche einzelnen Funktionen sind das und aus welchem Altsystem kommen sie?

======================================================================
======================================================================
END OF REPORT
======================================================================