======================================================================
DOCUMENT CHUNK REVIEW REPORT
======================================================================

Document ID:  c3659136-7b1a-432b-ada8-ff9273398918
Tenant ID:    a07960c9-32bc-5dba-a0bc-02a1b4fe1f2f
Schema:       rag
Generated:    2026-01-23T12:44:02.188613
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
  Mean Coherence:           72.0/100
  Mean Completeness:        68.3/100
  Mean Reference Resolution:74.7/100
  Mean Redundancy:          72.7/100
  Mean Overall:             71.9/100

  Lowest scoring chunks:
    - Chunk 88177e2c...: 61/100
      Reason: The chunk serves as a metadata header for a document chunk and is mostly coherent as a unit. It is s
    - Chunk 81452fef...: 71/100
      Reason: The chunk centers on vendor information and questions about data flow and system replacements; it is
    - Chunk 78952d83...: 84/100
      Reason: The chunk presents a coherent set of information-gathering questions about system usage and cloud/so

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
  1. Low average completeness (68.3). Chunks may need more context. Increase overlap or chunk size.

----------------------------------------------------------------------
CHUNK CONTENT
----------------------------------------------------------------------
Chunk 0 (ID: 88177e2c-9afe-5922-a704-268062a6bf87) - Length: 112
--------------------
Anlage 1 Systembeschreibung.pdf | Chunk 1 von 3
Anlage 1 KBV Rahmen IT 2.0 Systembeschreibung Systembeschreibung

======================================================================
Chunk 1 (ID: 78952d83-dbcd-51e2-b9b0-14f7bb13752f) - Length: 380
--------------------
Anlage 1 Systembeschreibung.pdf | Chunk 2 von 3
Allgemeine Informationen: Wo wird das System eingesetzt (Unternehmen)? In welchen Fachbereichen soll das System gesetzt werden? Wenn mehr als eine Software eingeführt wird: Welche Softwareanwendungen sind umfasst? (Hersteller und Produkte müssen benannt werden) 4. Cloudsoftware – SaaS, PaaS, IaaS? Wofür wird das System eingesetzt?

======================================================================
Chunk 2 (ID: 81452fef-3925-5f01-bc43-288c8660c7bc) - Length: 371
--------------------
Anlage 1 Systembeschreibung.pdf | Chunk 3 von 3
Anbieterseitige Informationen zum System internes Systemhandbuch (aktuelle Version), soweit vorhanden Welche Systeme bekommen Daten von diesem System? Wenn Altsysteme ersetzt werden: welche Systeme sind das? Wenn einzelne Funktionen ersetzt werden: welche einzelnen Funktionen sind das und aus welchem Altsystem kommen sie?

======================================================================
======================================================================
END OF REPORT
======================================================================