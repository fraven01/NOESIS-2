# Parser-Referenz

## PDF-Parser
- **Einsatzgebiet:** Extrahiert Seiteninhalte aus PDF-Dateien, erkennt Überschriften, Fließtext, Listen und tabellarische Ausschnitte für nachgelagerte Verarbeitung.
- **Erzeugte Artefakte:** Liefert Textblöcke mit Abschnittspfad, erkannte Tabellen als Zusammenfassungen samt Stichprobendaten sowie Bild-Assets mit Bounding-Box, Kontext vor/nach dem Bild und Sprachhinweisen.
- **Heuristiken & Schutzmaßnahmen:** Nutzt Schriftgrößen und Aufzählungsmarker für Struktur, kombiniert tabellarische Erkennung aus Layout- und Textanalysen, normalisiert Inhalte im sicheren Modus und kennzeichnet leere Seiten für optionales OCR; validiert Medien-Typen, entschlüsselt reparierte Dateien und lehnt verschlüsselte Payloads oder übergroße Assets ab.
- **Konfiguration:** Steuert Sicherheits-Normalisierung über `pdf_safe_mode`, aktiviert optionales OCR mit `enable_ocr` und nutzt `ocr_renderer` für Bild-Hooks.
- **Grenzen & Tests:** Erzwingt Einzel-Assets bis 25 MB, Gesamtbudget 200 MB und 512-Byte-Kontextfenster; Tests prüfen Abschnittserkennung, Tabellenauszüge, Asset-Kontexte, OCR-Auslösung, Fehlercodes sowie Spracherkennung auf kurzen Seiten.

## DOCX-Parser
- **Einsatzgebiet:** Liest Textverarbeitungspakete und wandelt sie in strukturierte Abschnitte, Listen und Tabellenstatistiken um.
- **Erzeugte Artefakte:** Erstellt Textblöcke für Überschriften, Fließtext und Listen inklusive hierarchischer Abschnittspfade sowie tabellarische Zusammenfassungen; exportiert eingebettete Bilder als Assets mit Binärdaten, Kontexten und Kandidaten für Beschriftungen.
- **Heuristiken & Schutzmaßnahmen:** Validiert Paketgröße und Eintragsanzahl, erkennt Überschriftenebenen und Listenstile, fasst Tabellen in Vorschauen zusammen und nutzt Alt-Text oder angrenzenden Text als Kontext; fehlende Referenzen oder zu große Bilder führen zu Fehlern.
- **Konfiguration:** Der Parser arbeitet ohne spezielle Schalter und folgt den Standardwerten der Pipeline-Konfiguration.
- **Grenzen & Tests:** Beschränkt Archive auf 10 000 Einträge, 500 MB unkomprimiert und 25 MB pro Asset; Tests decken Erkennung trotz generischer Medien-Typen, Tabellen-Metadaten, Alt-Text-Fallbacks, Kontexttrunkierungen sowie Fehlerfälle bei fehlenden oder übergroßen Ressourcen ab.

## PPTX-Parser
- **Einsatzgebiet:** Extrahiert Inhalte aus Präsentationen als Folien- und Notiztexte für Wissensaufbereitung.
- **Erzeugte Artefakte:** Gibt Textblöcke für Folien und optional Notizen mit Abschnittspfaden aus und liefert Bild- sowie Chart-Assets mit Kontexten, Alternativtexten, Notizen und Lokatoren.
- **Heuristiken & Schutzmaßnahmen:** Prüft Paketgrenzen, analysiert Formenrekursion, sammelt Text und Sprachhinweise pro Folie, verknüpft Alt-Text, Notizen und Folientitel, begrenzt Asset-Budgets und fügt Charts als referenzierbare Quellen hinzu.
- **Konfiguration:** `enable_notes_in_pptx` schaltet Notizerfassung zu, `emit_empty_slides` steuert Platzhalter für leere Folien.
- **Grenzen & Tests:** Setzt 25 MB pro Asset, 200 MB Gesamtbudget und validiert Präsentationsstruktur; Tests prüfen Notizumschaltung, Mehrfach-Notizen, Leerfolienverhalten, Asset-Budget-Fehler, Alt-Text-Kontexte, Chart-Quellen, Sprachlogik und Medien-Typ-Erkennung.

## Markdown-Parser
- **Einsatzgebiet:** Parst GitHub-kompatible Markdown-Dokumente inklusive Frontmatter, Listen und eingebettetem HTML.
- **Erzeugte Artefakte:** Produziert Textblöcke für Überschriften, Absätze, Listen, Code und tabellarische Zusammenfassungen; sammelt Bild-Assets aus Markdown und HTML mit Kontexten, Alt-Texten sowie Quellenhinweisen.
- **Heuristiken & Schutzmaßnahmen:** Entfernt Frontmatter, löst Fußnoten in Inline-Markierungen auf, verfolgt Überschriftenhierarchien, begrenzt Asset-Kontexte auf 512 Byte, ergänzt Quellen-URI und interpretiert HTML-Blöcke inklusive Tabellen.
- **Konfiguration:** Nutzt die Standard-Pipeline-Optionen ohne zusätzliche Schalter.
- **Grenzen & Tests:** Tabellen-Vorschauen und Code-Blöcke werden auf 500 Zeichen begrenzt, Assets bewahren Quellenkontext; Tests verifizieren Frontmatter-Normalisierung, Fußnoten, Checklisten, Tabellen-Metadaten, Sprachhinweise für Code sowie Bildextraktion aus Markdown und HTML mit Quellenverweisen.

## HTML-Parser
- **Einsatzgebiet:** Transformiert HTML-Seiten in lesbare Inhalte für Retrieval und Summarisierung.
- **Erzeugte Artefakte:** Liefert Textblöcke für Überschriften, Absätze, Listen, Tabellen und Code, ergänzt durch Bild-Assets mit Kontexten, Beschriftungskandidaten und Ursprungs-URI.
- **Heuristiken & Schutzmaßnahmen:** Nutzt optional eine Reader-Extraktion zur Reduktion von Boilerplate, wählt Hauptinhalte aus semantischen Containern, ignoriert Skripte und Navigation, fasst Tabellen zusammen und kombiniert Alt-Text oder Beschriftungen als Kontext.
- **Konfiguration:** `use_readability_html_extraction` aktiviert die vereinfachte Inhaltsauswahl.
- **Grenzen & Tests:** Kürzt Kontext auf 512 Byte, entfernt Navigation und Werbung, bewahrt Ursprungs-URI für Assets; Tests prüfen Varianten der Medien-Typ-Erkennung, Boilerplate-Reduktion, Tabellen- und Listeninformationen sowie Metadaten und Kontextkürzungen der extrahierten Assets.
