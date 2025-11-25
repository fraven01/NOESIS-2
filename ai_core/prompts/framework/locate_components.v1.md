# Baustein-Lokalisierung in Rahmenvereinbarung

## Ziel
Finde die vier Pflicht-Bausteine einer IT-Mitbestimmungs-Rahmenvereinbarung und kartiere, wo sie im Dokument stehen.

## Kontext
Du erhältst:
1. **Inhaltsverzeichnis** (ToC) des Dokuments (aus Überschriften-Hierarchie)
2. **Top-10 relevante Chunks** pro Baustein (aus semantischer Suche)

## Die vier Bausteine

**1. Systembeschreibung**
- Technische Beschreibung des IT-Systems
- Architektur, Module, Komponenten
- Typische Überschriften: "Systembeschreibung", "Technische Beschreibung", "Das System"

**2. Funktionsbeschreibung**
- Was macht das System? Welche Features?
- Use Cases, Prozesse
- Typische Überschriften: "Funktionsbeschreibung", "Funktionen", "Anwendungsbereiche"

**3. Auswertungen**
- Welche Berichte, Dashboards, Statistiken können erstellt werden?
- Oft in Anlagen/Unteranlagen strukturiert
- Typische Überschriften: "Auswertungen", "Berichte", "Reports", "Anlage X: Auswertungen"

**4. Zugriffsrechte**
- Wer darf was? Rollen, Berechtigungen
- Typische Überschriften: "Zugriffsrechte", "Berechtigungen", "Rollen", "Zugriffsregelungen"

## Aufgabe

Für **jeden der vier Bausteine**:

1. **Prüfe das ToC** nach passenden Überschriften (strukturelle Suche)
2. **Prüfe die Top-10 Chunks** nach semantischer Relevanz
3. **Bestimme die Location**:
   - `main`: Im Hauptdokument
   - `annex`: In einer Anlage
   - `annex_group`: Verteilt über mehrere (Unter-)Anlagen
   - `not_found`: Nicht im Rahmen geregelt

4. **Extrahiere Details**:
   - `outline_path`: Strukturpfad (z.B. "2", "§ 4", "Anlage 3")
   - `heading`: Die relevante Überschrift
   - `chunk_ids`: UUIDs der relevanten Chunks
   - `page_numbers`: Auf welchen Seiten?
   - `candidate_annex`: Falls in Anlage, welche? (z.B. "Anlage 1")

5. **Bewerte Confidence** (0.0-1.0):
   - 0.9-1.0: Sehr sicher (explizite Überschrift + passender Inhalt)
   - 0.7-0.8: Sicher (implizite Überschrift oder verstreuter Inhalt)
   - 0.5-0.6: Unsicher (nur vage Hinweise)
   - <0.5: Nicht gefunden

## Spezialfall: Anlagen-Gruppen

Bei Auswertungen oft:
```
Anlage 3: Auswertungen
  ├─ Anlage 3.1: Personalauswertungen
  ├─ Anlage 3.2: Leistungsauswertungen
  └─ Anlage 3.3: Verfügbarkeitsberichte
```

Gib an:
- `location`: "annex_group"
- `annex_root`: "Anlage 3"
- `subannex_pattern`: "3.x"
- `subannexes`: ["3.1", "3.2", "3.3"]

## Ausgabe (JSON)

```json
{
  "systembeschreibung": {
    "location": "main",
    "outline_path": "2",
    "heading": "2. Beschreibung des IT-Systems",
    "candidate_path": "2",
    "candidate_annex": null,
    "chunk_ids": ["uuid1", "uuid2"],
    "page_numbers": [2, 3],
    "confidence": 0.92,
    "evidence": {
      "structural": "Überschrift enthält 'Beschreibung' + 'System'",
      "semantic": "Chunks beschreiben technische Architektur",
      "parent_title": "§ 2 Systembeschreibung"
    }
  },
  "funktionsbeschreibung": {
    "location": "annex",
    "outline_path": "Anlage 1",
    "heading": "Anlage 1: Funktionsbeschreibung",
    "candidate_path": "A1",
    "candidate_annex": "Anlage 1",
    "chunk_ids": ["uuid5", "uuid6"],
    "page_numbers": [15, 16],
    "confidence": 0.88,
    "evidence": {...}
  },
  "auswertungen": {
    "location": "annex_group",
    "outline_path": "Anlage 3",
    "heading": "Anlage 3: Berichte und Auswertungen",
    "candidate_path": "Anlage 3",
    "candidate_annex": "Anlage 3",
    "annex_root": "Anlage 3",
    "subannex_pattern": "3.x",
    "subannexes": ["3.1", "3.2", "3.3"],
    "chunk_ids": ["uuid10", "..."],
    "page_numbers": [25, 26, 27],
    "confidence": 0.85,
    "evidence": {...}
  },
  "zugriffsrechte": {
    "location": "not_found",
    "outline_path": null,
    "confidence": 0.0,
    "searched_locations": ["main §4", "Anlage 2"],
    "recommendation": "Sollte in Anlage 2 ergänzt werden"
  }
}
```

## Wichtig
- **Nur kartieren**, nicht interpretieren oder umformulieren
- Wenn ein Baustein nicht gefunden: `location: "not_found"`, `confidence: 0.0`
- Bei Unsicherheit: `confidence < 0.7` + Begründung in `evidence`
- Mehrere Locations möglich: Wähle die vollständigste
