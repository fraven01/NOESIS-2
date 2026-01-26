======================================================================
DOCUMENT CHUNK REVIEW REPORT
======================================================================

Document ID:  47486796-abcf-4918-9fa5-db9ddfbf3d09
Tenant ID:    a07960c9-32bc-5dba-a0bc-02a1b4fe1f2f
Schema:       rag
Generated:    2026-01-25T16:36:59.815855
Total Chunks: 24

----------------------------------------------------------------------
CHUNK STATISTICS
----------------------------------------------------------------------
  Total characters: 33,527
  Min length:       265 chars
  Max length:       4697 chars
  Mean length:      1397 chars
  Median length:    991 chars
  Std deviation:    1105 chars

  Size distribution:
    tiny (<100)          0 
    small (100-300)      1 █
    medium (300-800)     7 ███████
    large (800-1500)     8 ████████
    huge (>1500)         8 ████████

----------------------------------------------------------------------
CONTEXTUAL ENRICHMENT
----------------------------------------------------------------------
  Enabled:            True
  Chunks with prefix: 24
  Missing prefix:     0
  Coverage ratio:     1.000
  Mean prefix length: 535
  Max prefix length:  668
  Samples:
    - This opening fragment serves as the header for the corporate works agreement. It identifies the two parties—Telefónica Deutschland Holding AG (the employer) and
    - Dieser Auszug stammt aus der KBV IT Rahmen 2.0 der Telefónica-Gruppe. Er markiert den Gegenstand der Vereinbarung: Festlegung der Verfahren und Grundsätze zur E
    - Contextual note: This excerpt is from the KBV IT Rahmen 2.0, in the scope section. It establishes the document’s boundaries for applying the Konzernbetriebsvere

----------------------------------------------------------------------
CHUNK BOUNDARY ANALYSIS
----------------------------------------------------------------------
  All chunk boundaries look clean!

----------------------------------------------------------------------
DETECTED PROBLEMS
----------------------------------------------------------------------
  Chunk 5:
    ⚠️ [too_long] Chunk is very long (3467 chars)
  Chunk 8:
    ⚠️ [too_long] Chunk is very long (2541 chars)
  Chunk 10:
    ⚠️ [too_long] Chunk is very long (4697 chars)
  Chunk 11:
    ⚠️ [too_long] Chunk is very long (2723 chars)
  Chunk 22:
    ⚠️ [too_long] Chunk is very long (2280 chars)
  Chunk 23:
    ℹ️ [incomplete_sentence] Chunk doesn't end with sentence punctuation

----------------------------------------------------------------------
LLM QUALITY EVALUATION
----------------------------------------------------------------------
  Mean Coherence:           79.5/100
  Mean Completeness:        73.3/100
  Mean Reference Resolution:71.5/100
  Mean Redundancy:          52.7/100
  Mean Overall:             69.2/100

  Lowest scoring chunks:
    - Chunk acc94cb6...: 34/100
      Reason: The chunk contains a fragment of a termination clause (Schlussbestimmung) but is interrupted by garb
    - Chunk 5f3adfc3...: 55/100
      Reason: The chunk presents a linked scope clause (spatial, personal, and substantive) that is mostly coheren
    - Chunk 5f7052c5...: 56/100
      Reason: The chunk forms a largely coherent single section about the introduction and use of evaluations, inc
    - Chunk d08458b6...: 58/100
      Reason: The chunk presents a coherent idea about AI-related negotiations and a goal to conclude a KBV within
    - Chunk 76332f39...: 60/100
      Reason: The text is a largely coherent legal rule about IT changes with clear sub-parts and exceptions. It r

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
  1. High variance in chunk lengths. Consider adjusting chunker settings for more consistent sizing.
  2. 8 very large chunks (>1500 chars) detected. Consider reducing max_chunk_size.

----------------------------------------------------------------------
CHUNK CONTENT
----------------------------------------------------------------------
Chunk 0 (ID: bdd1d62a-be88-5b5d-98b8-550751f188fb) - Length: 404
Contextual Prefix: This opening fragment serves as the header for the corporate works agreement. It identifies the two parties—Telefónica Deutschland Holding AG (the employer) and
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 1 von 24
Konzernb etriebsvereinbarung Rahmen IT 2.0 der Telefónica Deutschland Holding AG (im Folgenden: „ Arbeitgeber in “ oder „ Telefónica “ ) dem Konzernb etriebsrat der Telefónica Deutschland Holding AG (im Folgenden: „ Konzernbe triebsrat “ oder „ K BR “) Alle Parteien zusammen werden nachfolgend auch als „ Betriebsparteien “ bezeichnet.

======================================================================
Chunk 1 (ID: b151087b-3380-57ae-b65b-ece37fd9d561) - Length: 402
Contextual Prefix: Dieser Auszug stammt aus der KBV IT Rahmen 2.0 der Telefónica-Gruppe. Er markiert den Gegenstand der Vereinbarung: Festlegung der Verfahren und Grundsätze zur E
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 2 von 24
Gegenstand der Betriebsvereinbarung Mit dieser Konzernbetriebsvereinbarung (im Folgenden: „KBV“) vereinbaren die Parteien Verfahrensregeln und Grundsätze für die Einführung und Anwendung von technischen Einrichtungen, die dazu geeignet sind, das Verhalten oder die Leistung von Arbeitnehmern zu überwachen (nachfolgend „IT-Systeme“) .

======================================================================
Chunk 2 (ID: 5f3adfc3-73f5-5913-b29d-7aba53a73992) - Length: 576
Contextual Prefix: Contextual note: This excerpt is from the KBV IT Rahmen 2.0, in the scope section. It establishes the document’s boundaries for applying the Konzernbetriebsvere
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 3 von 24
Geltungsbereich Die KBV gilt räumlich für Telefónica und alle Betriebe aller mit ihr nach § 18 Abs. 1 AktG verbundenen Unternehmen . Diese Konzernbetriebsvereinbarung gilt persönlich für alle Mitarbeiter im Sinne von § 5 Abs. 1 BetrVG von Telefónica und der mit ihr nach § 18 Abs. 1 AktG verbundenen Unternehmen mit Ausnahme der Leitenden Angestellten im Sinne von § 5 Abs. Die KBV gilt sachlich für die Einführung und Anwendung von IT-Systemen, soweit eine Zuständigkeit des KBR gem. 1 BetrVG begründet ist.

======================================================================
Chunk 3 (ID: fea25ae2-3e44-5b10-b508-be1be6a96714) - Length: 1712
Contextual Prefix: Dieser Abschnitt dient der Einordnung zentraler Begriffe im IT-Rahmen der Konzernbetriebsvereinbarung IT 2.0. Er definiert, was als IT-System gilt, welche perso
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 4 von 24
Definitionen IT-System: technische Einrichtung, die dazu geeignet ist, das Verhalten oder die Leistung von Arbeitnehmern zu überwachen (auch kurz „System“). Personenbezogene Mitarbeiterdaten: Personenbezogene Daten, die Rückschlüsse auf Verhalten- und/oder Leistung von Mitarbeitern ermöglichen . Auswertung: Sichtung, Sortierung, Zusammenstellung oder Inbeziehungsetzen von Daten , ggf. mit anderen Daten, inner- oder außerhalb des Systems, soweit damit ein Rückschluss auf Leistung oder Verhalten eines Mitarbeiters oder einer Gruppe von weniger als sechs Mitarbeitern möglich ist. Sofern die Auswertung Informationen über Leistung oder Verhalten einer Gruppe von sechs oder mehr Mitarbeitern enthält, aber Rückschlüsse auf die Leistung oder das Verhalten der Person ermöglicht, die die Gruppe fachlich verantwortet, handelt es sich ebenfalls um eine Auswertung im Sinne dieser KBV . Admin i strator : Person, die von Telefónica autorisiert ist, administrative Aufgaben in einem System auszuführen . Administrative Aufgaben: Verwalten von Benutzerkonten, Überwachen der Systemleistung, Installieren und Aktualisieren von Software, Einrichten und Verwalten von Sicherheitsrichtlinien, Analysieren und Beheben von technischen Problemen und funktionale Bereitstellung von Softwarefunktionen an die Benutzer des Systems. Mitarbeitergespräch: Mitarbeitergespräche in diesem Sinne sind zielorientierte Gespräche zwischen einer Führungskraft und einem oder mehreren Mitarbeitern, in denen die Leistung und/oder das Verhalten des Mitarbeiters besprochen wird und dem Mitarbeiter ggf. Weisungen zur Änderung von Verhalten oder Leistung erteilt werden.

======================================================================
Chunk 4 (ID: 9b942e96-c7c3-5642-a6f9-cb0a9efa73cd) - Length: 805
Contextual Prefix: Dieser Ausschnitt gehört zum KBV IT Rahmen 2.0 der Telefónica Deutschland Holding AG. Es handelt sich um den Abschnitt zur Einführung von IT-Systemen (Chunk 5 v
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 5 von 24
Einführung von IT-Systemen Die Einführung von IT-Systemen erfolgt jeweils nach Abschluss einer gesonderten Konzernbetriebsvereinbarung (nachfolgend auch „Einzel-Konzernbetriebsvereinbarung“ oder „Einzel-KBV“). Hierfür wird das Muster gemäß Anlage 0 verwendet. Die Einführung eines IT-Systems ohne oder vor Abschluss einer solchen Einzel-KBV zur Regelung der Einführung und Anwendung des IT-Systems ist unzulässig . Die Einführung des gesamten IT-Systems kann im Einzelfall in unterschiedliche Teile aufgeteilt werden, die in unterschiedlichen Einzel-Konzernbetriebsvereinbarungen in verschiedenen Schritten sukzessive geregelt werden können, zum Beispiel Teil des Gesamtsystems, den die User sehen Reportingtools Datenmanagementsysteme .

======================================================================
Chunk 5 (ID: 8c07cac1-a7c6-5ece-ab67-512ec967df97) - Length: 3467
Contextual Prefix: Context: This excerpt is from the Informationspflichten section of Telefónica’s Konzernbetriebsvereinbarung IT Rahmen 2.0. It details the KBR information obliga
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 6 von 24
Informationspflichten Der KBR wird spätestens mit einem Vorlauf von 10 Arbeitstagen vor der KBR-Sitzung, für die d er Arbeitgeber beantragt hat, den Gegenstand der Einführung des neuen IT-Systems zur Beratung auf die Tagesordnung zu setzen, über die beabsichtigte Einführung des neuen IT-Systems informiert . Der KBR ist berechtigt, eine Expertengruppe zu bestimmen, die ihn bei der Ausübung seiner Mitbestimmung zur Einführung und Anwendung neue r IT-System e unterstützt und berät. Die Expertengruppe besteht aus den Sprechern des KISA und KCSA, dem PMO und einem Mitglied des KoBA . Die Expertengruppe kann bei Erforderlichkeit sachkundige Mitglieder eines Betriebsratsgremiums hinzuziehen, ohne dass eine vorherige Zustimmung von Telefónica erforderlich ist. Die Expertengruppe teilt Telefónica unverzüglich mit, welche Personen hinzugezogen werden. Die sach kundigen Betriebsratsmitglieder sind für die Tätigkeit in der Expertengruppe unter Fortzahlung ihrer Vergütung von ihren arbeitsvertraglich geschuldeten Tätigkeiten freigestellt . Die Information muss eine System beschreibung ( Anlage 1 ) und Funktionsbeschreibung ( Anlage 2 ) enthalten. Außerdem übersendet d ie Arbeitgeber in dem KBR zeitgleich den Entwurf einer Einzel-Konzernbetriebsvereinbarung für das betreffende IT-System. Hierfür wird das Muster gemäß Anlage 0 verwendet. Der KBR gibt nach Überreichung der Unterlagen und vor der KBR-Sitzung an Telefónica die Rückmeldung, ob die Informationen aus den Unterlagen nach dem derzeitigen Kenntnisstand vollständig und ausreichend sind und/oder, ob weitere Informationen benötigt werden und wenn ja, welche. Verlangt der KBR oder die Expertengruppe weitere Informationen, präsentiert und erläutert d ie Arbeitgeber in der Expertengruppe das IT-System vor der vorgenannten KBR-Sitzung. Dieser Termin findet grundsätzlich 5 Arbeitstage vor der kommenden KBR-Sitzung statt. Sofern ein Testsystem vorhanden ist, findet auf Wunsch der Expertengruppe eine Systemdemonstration im Testsystem statt. Der KBR gibt der Arbeitgeberin innerhalb von 5 Arbeitstagen (Montag bis Freitag) nach der KBR-Sitzung eine Rückmeldung und teilt mit , ob der KBR dem Abschluss der KBV zugestimmt hat oder d er KBR den Entwurf der KBV geändert und dem Abschluss der geänderten KBV zugestimm t hat oder d er KBR noch keinen Beschluss zum Abschluss einer KBV zur Einführung des IT-Systems gefasst hat, weil der KBR vorher prüfen möchte, ob andere Mitbestimmungsrechte außer § 87 Abs. 6 BetrVG durch die Einführung des IT-Systems tangiert sind oder andere Vorkehrungen zum Schutz der Mitarbeiter vor Einführung des Systems erforderlich sind . der KBR Bedenken bezgl. der Einführung des IT-Systems an sich hat. der KBR Bedenken bzgl. der konkreten Anwendung des Systems hat . der KBR weitere Informationen benötigt, bevor er einen Beschluss fassen kann . Wenn der KBR keinen Beschluss aus einem der in a. genannten Gründe fasst , beschließt er eine Verhandlungsgruppe und nennt sie de r A rbeitgeberin . Im Falle von d. nennt der KBR die noch erforderlichen Informationen. Telefónica liefert dem KBR diese schnellstmöglich, spätestens 10 Tage vor der nächsten KBR-Sitzung. Der KBR gibt nach Überreichung der Unterlagen wiederum vor der KBR-Sitzung an Telefónica die Rückmeldung, ob die Informationen nunmehr vollständig und ausreichend sind und/oder, ob weitere Informationen benötigt werden und wenn ja, welche.

======================================================================
Chunk 6 (ID: 9191e882-1b63-5a5a-803e-b2216c6395ce) - Length: 1037
Contextual Prefix: Dieser Ausschnitt gehört zum Abschnitt des IT-Rahmens 2.0, der den Grundsatz festhält, dass die Einführung eines IT-Systems erst nach Abschluss der jeweiligen E
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 7 von 24
Kein System ohne KBV Sofern der KBR dem Abschluss der Einzel-Konzernbetriebsvereinbarung für das betreffende IT-System zustimmt, darf das IT-System nach Abschluss der betreffenden KBV eingeführt werden. Auswertungen dürfen nur unter Einhaltung der Regelungen in Ziffer 8 eingeführt und angewendet werden. A bweichend hiervon ist die Anwendung eines I T -Systems erlaubt, ohne dass zuvor eine KBV abgeschlossen werden muss, so weit die Anwendung darauf beschränkt ist, das System zu testen und zu evaluieren, um dessen Eig nung für den Nutzungszweck zu prüfen. Echtdaten von Mitarbeitern dürfen hierfür nicht mit dem System v erarbeitet werden . Ausgenommen hiervon sind Login daten d er Nutzer , die für das Testing und die Evaluierung zwingend erforderlich sind. Auswertungen von Leistungs- oder Verhaltensdaten der Nutzer sind verboten. Nach Abschluss dieser Prüfung ist eine weitere Anwendung des Systems nur zulässig, wenn zuvor eine KBV hierzu abgeschlossen wurde.

======================================================================
Chunk 7 (ID: e0951a15-b63d-5227-9458-e03052ebb4d9) - Length: 838
Contextual Prefix: Dieser Abschnitt regelt die Einführung und Anwendung von Funktionen des IT-Systems nach dem Abschluss der Einzel-Konzernbetriebsvereinbarung. Er setzt voraus, d
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 8 von 24
Einführung und Anwendung von Funktionen Nach Abschluss der Einzel-Konzernbetriebsvereinbarung werden nur diejenigen Funktionen des Systems technisch eingeführt und angewendet, für die in der Anlage 2 in der Spalte „Einsatzweise bei Telefónica: Soll die Funktion verwendet werden?“ ein „Ja“ angegeben ist. Die Einführung und Anwendung weiterer Funktionen des Systems ist jeweils erst zulässig, nachdem sich die Betriebsparteien hierauf geeinigt und die Anlage 2 der betreffenden Einzel-Konzernbetriebsvereinbarung entsprechend angepasst haben. Für die Funktionen, die danach eingeführt und angewendet werden dürfen, gelten die ggf. in der jeweiligen Einzel-Konzernbetriebsvereinbarung festgelegten technischen bzw. organisatorischen Einschränkungen und Verwendungszwecke.

======================================================================
Chunk 8 (ID: 5f7052c5-7ce8-5a27-aba7-9b024ed56e98) - Length: 2541
Contextual Prefix: Dieser Abschnitt gehört zur Konzernbetriebsvereinbarung Rahmen IT 2.0 und behandelt die Einführung und Anwendung von Auswertungen. Er regelt den zulässigen Abla
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 9 von 24
Einführung und Anwendung von Auswertungen Die Einführung und Anwendung einer Auswertung im Sinne dieser Konzernbetriebsvereinbarung ist jeweils erst nach Abschluss des nachfolgend beschriebenen Ablaufs zulässig und im Übrigen untersagt. Anwendung durch Administratoren: Voraussetzung für die Erteilung von Zugriffsrechten für Administratoren auf Auswertungen , die noch nicht als Anlage zu der betreffenden Einzel-Konzernbetriebsvereinbarung geregelt sind, ist neben dem Abschluss der betreffenden Einzel-Konzernbetriebsvereinbarung, eine vollständig e Dokumentation der Auswertung mit allen Informationen gemäß Anlage 4 . Die Dokumentation muss revisionssicher an einem Ort erfolgen, zu dem der KBR und die von ihm benannte Expertengruppe jederzeit Zugang haben. Der KBR muss gesondert darüber informiert werden, dass und wo der Arbeitgeber die Dokumentation abgelegt hat. Ab dem Zeitpunkt, zu dem der KBR über die Dokumentation informiert ist und ihm die Zugangsmöglichkeiten zur Verfügung stehen, dürfen Administratoren Auswertungen unter den vorgenannten Voraussetzungen ausschließlich zur Erledigung von administrativen Aufgaben gemäß Ziffer 3 e ) verwenden . Der KBR hat einen Durchführungsanspruch entsprechend § 77 Abs. 1 BetrVG dahingehend, dass die Auswertung durch Administratoren nur zur Erledigung administrativer Aufgaben angewendet werden darf. Verboten ist insbesondere eine Weitergabe der mit einer Auswertung erlangten Informationen über eine Leistung oder ein Verhalten eines Arbeitnehmers oder einer Gruppe von weniger als sechs Arbeitnehmern an andere Personen. Eine hiervon abweichende Einführung und Anwendung der Auswertung stellt einen Verstoß gegen die zur Einführung und Anwendung des betreffenden IT-Systems abgeschlossene Einzel-KBV dar. Anwendung durch andere Personen: Andere Personen als Administratoren dürfen eine Auswertung anwenden, nachdem sie in der betreffenden Einzel- KBV gemäß Anlage 3 (Anlage 3 der betreffenden Einzel-KBV ) geregelt und deren Anwendung durch die betreffende Personengruppe gestattet ist. Für die in der Anlage 5 genannten Zwecke und Maßnahmen dürfen Auswertungen sowie durch deren Anwendung erlangte Informationen nur dann verwendet werden, wenn dies für die betreffende Auswertung in der Anlage 3 der betreffenden Einzel-KBV ausdrücklich geregelt ist. Alle Erleichterungen gemäß Ziffer 8 a) und b) und Ziffer 9 gelten nicht für die Verwendung von Auswertungen für die in Anlage 5 genannten Zwecke und Maßnahmen.

======================================================================
Chunk 9 (ID: bedaa611-d003-51b0-9fcd-41d932e348e2) - Length: 1994
Contextual Prefix: Dieser Abschnitt ist Teil der Regelungen zu Auswertungen innerhalb der KBV. Er beschreibt den befristeten Verzicht auf Unterlassungsansprüche, der sechs Monate 
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 10 von 24
Befristeter Verzicht auf Unterlassungsanspruch Zur Vorbereitung der Regelung von Auswertungen gemäß Ziffer 8 b ) füllt Telefónica d en Fragebogen gemäß Anlage 6 aus. Sofern eine Frage des Teils 1 des Fragebogens mit „ ja “ beantwortet wurde, muss auch Teil 2 des Fragebogens vollständig beantwortet und revisionssicher dokumentiert werden. Der KBR verzichtet für einen Zeitraum von 6 Monaten, beginnend mit Zugang der Information gemäß Anlage 6 Teil 1 und 2 für die jeweilige Auswertung auf die Geltendmachung eines Unterlassungsanspruchs, soweit die Auswertung ausschließlich so an ge wende t wird , wie in der Dokumentation angegeben. Eine Verwendung der Auswertung bzw. der mit der Auswertung erlangten Informationen für in Anlage 5 genannte Zwecke und Maßnahmen ist nicht vom Verzicht des Unterlassungsanspruches umfasst. Die Parteien sollen innerhalb dieses 6-Monats- Zeitraums eine abschließende Regelung gemäß Anlage 3 ( Anlage 3 der betreffenden Einzel-Konzernbetriebsvereinbarung) für die Anwendung der Auswertung außerhalb des in Ziffer 8 a) geregelten Anwendungsbereiches vereinbaren. Der Verzicht auf den Unterlassungsanspruch des KBR erlischt, wenn bis zum Ablauf des 6-Monats-Zeitraums eine solche Regelung nicht einvernehmlich zwischen den Parteien oder durch den Spruch einer Einigungsstelle getroffen wurde. In diesem Fall ist die Arbeitgeberin verpflichtet, allen Personen, mit Ausnahme der Administratoren, das Zugriffsrecht auf die Auswertung zu entziehen. Stellt der KBR oder Telefónica während des 6-Monatszeitraums fest, dass Auswertungen abweichend von der Dokumentation der Auswertung angewendet oder ohne Gestattung zu einem in Anlage 5 genannten Zweck verwendet werden ("Verstoß"), sind beide Betriebsparteien berechtigt, das Verfahren gemäß Ziffer 22 dieser Konzernbetriebsvereinbarung einzuleiten. Die Regelungen gemäß Ziffer 22 dieser Konzernbetriebsvereinbarung finden in diesem Fall Anwendung.

======================================================================
Chunk 10 (ID: de5b0fbd-47ff-5aa5-aecc-135369b758c9) - Length: 4697
Contextual Prefix: Context: This is from the “Allgemeine Vorgaben zur Verwendung von geregelten Auswertungen” section of the Konzernbetriebsvereinbarung IT Rahmen 2.0. The passage
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 11 von 24
All gemeine Vorgaben zur Verwendung von geregelten Auswertungen Auswertungen, die gemäß Anlage 3 ( Anlage 3 der betreffenden Einzel- KBV) geregelt und gestattet sind, dürfen über die in der jeweiligen Einzel-KBV zum konkreten IT-System hinaus geregelten Zwecke und Maßnahmen, generell verwendet werden für: die Ausübung des Direktionsrechts, Mitarbeitergespräche sowie die Ableitung und Begründung personeller Einzelmaßnahmen, sofern die nachfolgenden Regelungen eingehalten werden. Auswertungen, die nicht in einer Einzel-KBV geregelt sind, dürfen für diese Zwecke nicht verwendet werden. Verwendung zur Ausübung des Direktionsrechts: Unzulässig sind Weisungen von Telefónica (insbesondere von Führungskräften), mit denen Mitarbeiter aufgefordert werden, Informationen über die eigene Person oder andere Mitarbeiter über die eigene Leistung oder Verhalten oder anderer Mitarbeiter aus denen sich mittelbar die eigene Leistung oder Verhalten oder anderer Mitarbeiter ergibt vorzulegen, auf die die anweisende Person (z. Führungskraft) selbst keinen Zugriff in einem IT-System hat. Ebenso unzulässig sind Weisungen der Führungskraft, Informationen über die eigene Leistung oder Verhalten vorzulegen, die die Führungskraft aufgrund ihrer Rechte im System selbst direkt aus dem System erhalten kann. Verwendung für Mitarbeitergespräche: Für Mitarbeitergespräche gelten die nachfolgenden Bestimmungen: Leistungen oder Verhalten einzelner Mitarbeiter oder von Mitarbeitergruppen mit weniger als sechs Mitarbeitern dürfen in Mitarbeitergesprächen und Teammeetings nur unter Beachtung der gesetzlichen, datenschutzrechtlichen Regelungen und der Persönlichkeitsrechte von Mitarbeitern besprochen werden. Die Verwendung der jeweiligen Auswertung , für die in der Anlage 3 der betreffenden Einzel KBV geregelten Zwecke wird hierdurch nicht eingeschränkt . Vor einem Mitarbeitergespräch sind dem Mitarbeiter folgende Informationen mit ausreichendem zeitlichem Vorlauf, d. grundsätzlich zwei Arbeitstage (wobei Samstag und Sonntag nicht als Arbeitstage in diesem Sinne zählen) im Voraus, zur Kenntnis zu geben: Zweck und Ziel des Gespräches Von der Führungskraft betrachtete Leistungs- bzw. Verhaltensdaten sowie die von der Führungskraft hieraus gewonnenen Erkenntnisse. Informationen auf der Grundlage von Auswertungen dürfen seitens Telefónica maximal innerhalb von 30 Tagen 6-mal zum Inhalt von Mitarbeitergesprächen zur Leistungs- und Verhaltenskontrolle gemacht werden. F ür Mitarbeitergespräche über Terminvereinbarungen Besuchsauswertungen Telefonakquise (Kaltakquise) Besuchsfrequenz ordnungsgemäßes Pflegen der Einträge in IT-Systemen gelten ergänzend folgende Regelungen: Sie sind nur zulässig, wenn dem Mitarbeiter dieses Thema ausdrücklich im Vorfeld als Zweck und Ziel des Gespräches mitgeteilt wurde. In einem Mitarbeitergespräch dürfen maximal zwei der vorgenannten Themen besprochen werden. Sofern Mitarbeitergespräche zu einem der vorgenannten Themen geführt werden, müssen sowohl die Position der Führungskraft als auch die des Mitarbeiters besprochen und dokumentiert werden. Mitarbeitergespräche zu einem der vorgenannten Themen sollen an einem auf einen arbeitsfreien Tag folgenden Tag nicht vor 12 Uhr stattfinden. Die Betriebsparteien können für bestimmte Fachbereiche bzw. Teams von den Regelungen dieses Absatzes (iii) a bweichende Regelungen vereinbaren . Etwaige sonstige Betriebsvereinbarungen oder Regelung en zu Mitarbeitergesprächen bleiben von dieser Regelung unberührt und sind einzuhalten. Ver wendung für personelle Maßnahmen : Auswertungen , die in einer Einzel-KBV explizit geregelt sind, dürfen zur Ableitung und Begründung personeller Maßnahmen herangezogen werden. Kündigungen, die aus der Anwendung von Auswertungen oder Funktionen bzw. daraus resultierender Informationen abgeleitet oder damit begründet werden, bedürfen der vorherigen Zustimmung des Betriebsrats (§ 102 Abs. Dies gilt nicht für Kündigungen, die mit einer Straftat begründet werden, wegen derer in z ulässiger Weise eine Auswertung gemäß Ziffer 11 stattgefunden hat. Zuständig für die Zustimmung ist der örtliche Betriebsrat, der für die Mitbestimmung bei der betreffenden Kündigung gemäß § 102 BetrVG zuständig ist. Bei Meinungsverschiedenheiten gilt das in § 102 Abs. 6 BetrVG vorgesehene Verfahren. Eine ohne Zustimmung des Betriebsrats ausgesprochene Kündigung ist unwirksam. Ausgenommen hiervon sind Kündigungen , die ausschließlich aus betrieblichen Gründen erfolgen (betriebsbedingte Kündigungen). Bezüglich Ziffer d ) und e ) sind ausschließlich Systeme gemeint, die im Rahmen der KBV Rahmen-IT 2.0 eingeführt worden sind.

======================================================================
Chunk 11 (ID: 6e4a2b92-c60b-5d2a-b0de-6697a03d49be) - Length: 2723
Contextual Prefix: Der Chunk gehört zum Kapitel Auswertungen im IT-Rahmen der Konzernbetriebsvereinbarung IT 2.0. Er definiert die Voraussetzungen und zulässigen Zwecke der Verarb
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 12 von 24
Auswertung zur Aufdeckung von Straftaten Zur Aufdeckung von Straftaten dürfen personenbezogene Mitarbeiterdaten nur dann verarbeitet und ausgewertet werden, wenn (1) zu dokumentierende tatsächliche Anhaltspunkte den Verdacht begründen, dass ein Mitarbeiter im Kontext des Beschäftigtenverhältnisses eine Straftat begangen hat, (2) die Verarbeitung zur Aufdeckung erforderlich ist und (3) das schutzwürdige Interesse des Mitarbeiters an dem Ausschluss der Verarbeitung nicht überwiegt, insbesondere Art und Ausmaß im Hinblick auf den Anlass nicht unverhältnismäßig sind . Bestätigt das Ergebnis einer solchen Auswertung den Verdacht, dürfen die Ergebnisse nur zur weiteren Aufdeckung und Verfolgung der betreffenden Straftat(en) verwendet werden. Dokumentiert das Ergebnis einer solchen Auswertung einen konkreten Verdacht bzgl. einer anderen Straftat, kann das Ergebnis, sofern die übrigen Voraussetzungen von Ziff. 1 1 a) erfüllt sind, ausschließlich für die weitere Verfolgung dieser anderen Straftat(en) verwendet werden. Der jeweils für den Mitarbeiter zuständige Betriebsrat ist vor einer solchen Auswertung zumindest in Textform über (1) die Quelle (ohne Nennung personenbezogener Daten von etwaigen Hinweisgebern) für den Sachverhalt, aus dem die tatsachlichen Anhaltspunkte resultieren, (2) den begründeten Verdacht, (3) die Erforderlichkeit der Auswertung zur Aufdeckung der Straftat und (4) die Verhältnismäßigkeit der Auswertung zu informieren. Dem zuständigen Betriebsrat muss die Möglichkeit eingeräumt werden, an der Auswertung teilzunehmen. Hierbei ist auszuschließen, dass der Betriebsrat Einsicht in Kundendaten erhält. Ist der Mitarbeiter schwerbehindert oder gleichgestellt, so ist zusätzlich die zuständige Schwerbehindertenvertretung vor der geplanten Auswertung zu informieren. Ist der Betroffene Auszubildender, so ist zusätzlich die zuständige Jugend- und Auszubildendenvertretung zu informieren. Sollte dem zuständigen Betriebsrat die Teilnahme an der Auswertung nicht möglich sein, informiert der Auswertende den zuständigen Betriebsrat über das Ergebnis der Auswertung. Wird der Verdacht der Straftat durch die Auswertungsergebnisse nicht bestätigt, sind die Auswertungsergebnisse und die Dokumentationen von allen Beteiligten sofort zu löschen. Etwaige Ermittlungsberichte enthalten ausschließlich anonymisierte Auswertungsergebnisse. Sofern sich der Verdacht aufgrund der Auswertungsergebnisse bestätigt hat, werden diese und die Dokumentationen gelöscht, sobald sie zur Aufdeckung und Feststellung der Straftaten bzw. zur Beweisführung im Rahmen sich hieran anschließender rechtlicher Verfahren nicht mehr erforderlich sind.

======================================================================
Chunk 12 (ID: 76332f39-d4ac-5771-8d4d-a85023f73420) - Length: 945
Contextual Prefix: Teil dieses Dokuments befasst sich mit Änderungsprozessen von IT-Systemen und zugehörigen Auswertungen im Rahmen der Konzernbetriebsvereinbarung KBV IT Rahmen 2
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 13 von 24
Änderung von IT-Systemen Änderungen von Funktionen gemäß Anlage 2 sind erst zulässig , nachdem sie in der betreffenden Einzelkonzernbetriebsvereinbarung vereinbart sind. Für Änderungen von Auswertungen gelten die Regelungen gemäß den Ziffer n 8 bis 10 entsprechend: Änderungen von Auswertungen sind grun d sätzlich zulässig, nachdem sie in der betreffenden Einzelkonzernbetriebsvereinbarung ( Anlage 3 der Einzel-KBV) vereinbart sind. Abweichend hiervon, dürfen Administratoren geänderte Auswertungen zu administrativen Zwecken verwenden, nachdem das Verfahren gemäß Ziffer 8 a ) abgeschlossen ist. Für a lle a ndere n Personen gilt auch bei Änderungen von Auswertungen der befristete Verzicht auf den Unterlassungsanspruch gemäß Ziffer 9 mit der Maßgabe, dass der 6-Monatszeitraum mit Zugang der Information zu der betreffenden Auswertung gemäß Anlage 6 Teil 1 und 2 beginnt.

======================================================================
Chunk 13 (ID: 8e78fcc1-6f05-5c66-97ba-95c6e386d3f3) - Length: 1413
Contextual Prefix: Dieser Abschnitt gehört zum Auditkapitel des KBV IT Rahmen 2.0. Er beschreibt das Prüfungs- und Kontrollrecht des Konzernbetriebsrats (KBR): Der KBR darf eingef
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 14 von 24
Auditregelung Der KBR ist jederzeit berechtigt, die eingeführten Systeme und deren Anwendung dahingehend zu überprüfen, ob die in den Einzel-Konzernbetriebsvereinbarungen und in dieser Rahmenkonzernbetriebsvereinbarung getroffenen Regelungen eingehalten werden. Zur Überprüfung der Anwendung der eingeführten IT-Systeme sind für die Mitglieder der Gruppe, denen die Aufgabe der Überprüfung übertragen wurde, Systemzugänge einzurichten oder Demonstrationen im Echtsystem vorzuführen. Der KBR hat das Recht, jederzeit die Systemdokumentation für die einzelnen IT-Systeme einzusehen. Dem KBR sind u.a. alle Einsichten zu gewähren und Informationen zu erteilen, die er benötigt, um überprüfen zu können, ob die Berechtigungen für Auswertungen und die Anwendung von Funktionen in IT-Systemen ausschließlich so vergeben sind, wie in den Einzelkonzernbetriebsvereinbarungen geregelt. Der KBR kann sich nach Absprache auch an sonstigen vereinbarten Überprüfungen beteiligen. Die Einrichtung der Systemzugänge, Vorführung von Demonstrationen im Echtsystem und Gewährung von Einsichten sind dem KBR bzw. der von ihm benannten Gruppe innerhalb von 14 Tagen ab dem Zeitpunkt der Anforderung durch den KBR zu geben. Bei allen Überprüfungen ist ein Zugriff auf Kundendaten auszuschließen, soweit die Überprüfung auch ohne Zugriff auf Kundendaten möglich ist.

======================================================================
Chunk 14 (ID: d08458b6-ea47-5cf3-aafc-eb6d9634eaea) - Length: 265
Contextual Prefix: Dieser Abschnitt gehört zur Konzernbetriebsvereinbarung IT Rahmen 2.0 und behandelt Künstliche Intelligenz. Er präzisiert, dass die Betriebsparteien gesonderte 
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 15 von 24
Künstliche Intelligenz Die Parteien erklären , Verhandlungen über gesonderte Regelungen zu KI auf zu nehmen . Die Parteien haben das Ziel , hierzu eine KBV innerhalb von 12 Monaten abzuschließen .

======================================================================
Chunk 15 (ID: 6f81df0b-b7f8-54cf-88c0-b0715047e20d) - Length: 492
Contextual Prefix: Contextual framing: This fragment sits in the information and training obligations section of the KBV Rahmen IT 2.0. It states that Telefónica must inform all e
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 16 von 24
Hinweispflichten der Arbeitgeberin zur KBV Rahmen IT Telefónica ist verpflichtet, alle Mitarbeiter über die bestehenden Rechte und Pflichten dieser KBV im Rahmen der regelmäßig stattfindenden Compliance-Schulungen zu informieren. Diese werden bevorzugt als webbas e d Trainings durchgeführt. Telefónica wird dem KBR die Trainingsinhalte vorab zur Prüfung vorlegen und Hinweise und Änderungswünsche des KBR hierzu beachten .

======================================================================
Chunk 16 (ID: b1742c5c-57df-5061-91ed-6a2f2ad6b970) - Length: 824
Contextual Prefix: Dieser Abschnitt gehört zum Informations- und Schulungsmantel der KBV IT Rahmen 2.0. Er regelt, dass vor der Einführung eines IT-Systems alle betroffenen Mitarb
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 17 von 24
Information der Mitarbeiter und Schulungsmaßnahmen über Einzel-Konzernbetriebsvereinbarungen Vor Einführung eines IT-Systems werden alle Mitarbeiter, die eine Berechtigung für das System erhalten sollen, sowie alle Mitarbeiter, deren personenbezogene Mitarbeiterdaten mit dem System verarbeitet werden, über den Inhalt und das Inkrafttreten der Einzel-KBV zum konkreten System, über die Funktionen und Auswertungen des Systems und darüber informiert, inwieweit und wozu sie diese verwenden dürfen. Die im Falle der Einführung oder Änderung eines Systems erforderlichen Schulung en bzw. Einweisung en werden seitens des Arbeitgebers rechtzeitig, d.h. bevor dem betreffenden Mitarbeiter eine Rolle für das jeweilige IT-System zugewiesen wird, durchgeführt .

======================================================================
Chunk 17 (ID: 06c3067b-a9f0-5aa7-97fc-b394d9aec496) - Length: 580
Contextual Prefix: Diese Passage gehört zum Bereich Datenschutzrechte der KBV IT Rahmen 2.0. Sie fasst zusammen, dass Mitarbeiter jederzeit ihre nach DS-GVO geltenden Rechte bei d
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 18 von 24
Rechte der Mitarbeiter Mitarbeiter können jederzeit die ihnen zustehenden Rechte nach Art. DS-GVO bei der zuständigen Personalabteilung geltend machen. Die in der DS-GVO und im BDSG geregelten Einschränkungen oder Ausschlüsse werden durch diese Regelung nicht ausgeschlossen. Mitarbeiter haben das Recht, sich jederzeit an den betrieblichen Datenschutzbeauftragten und/oder den zuständigen Betriebsrat in datenschutzrechtlichen Angelegenheiten zu wenden. Daraus dürfen dem Mitarbeiter keine Nachteile entstehen.

======================================================================
Chunk 18 (ID: 59d01853-2e30-50a3-8330-05b30719b1af) - Length: 1100
Contextual Prefix: Kontextualisierung: Dieser Fragment gehört zu einem Abschnitt der Konzernbetriebsvereinbarung IT Rahmen 2.0 und fasst die Rechte des Konzernbetriebsrats (KBR) i
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 19 von 24
Rechte des KBR Sämtliche Informationen, die dem KBR nach dieser KBV zu erteilen sind, haben in deutscher Sprache zu erfolgen. Sofern die Informationen in einer anderen Sprache erfolgen, hat der KBR das Recht, diese Informationen – soweit erforderlich – auf Kosten des Arbeitgebers in die deutsche Sprache übersetzen zu lassen. Der KBR kann bei der Durchführung seiner Aufgaben nach näherer Vereinbarung mit Telef ónica auf Kosten von Telefónica Sachverständige hinzuziehen, sofern dies zur ordnungsgemäßen Erfüllung seiner Aufgaben erforderlich ist. Es gilt § 80 Abs. Für die Geheimhaltungspflicht des Sachverständigen gilt § 79 BetrVG entsprechend. Der KBR kann jederzeit verlangen, dass ihm sachkundige Mitarbeiter als Auskunftsperson zur Verfügung gestellt werden. Der KBR hat jederzeit das Recht, den betrieblichen Datenschutzbeauftragten und weitere interne Sachverständige hinzuzuziehen und soll diese bevorzugt als interne Sachverständige in Anspruch nehmen , sofern deren Hinzuziehung zur Klärung der Frage ausreichend ist.

======================================================================
Chunk 19 (ID: 2afbc226-3602-5740-a793-1cc0ae16534d) - Length: 1313
Contextual Prefix: Der vorliegende Abschnitt gehört zum IT-Rahmen der KBV und behandelt Grundsätze zur Vergabe von Berechtigungen. Kernziel ist die Gewährleistung minimaler Zugrif
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 20 von 24
Grundsätze zur Vergabe von Berechtigungen Berechtigungen werden so vergeben, dass Mitarbeiter genau die Zugriffsrechte erhalten, die der Mitarbeiter zwingend benötigt, um seine Aufgaben erfüllen zu können. Die Vergabe der Berechtigungen erfolgt über die Zuweisung von Rollen an die Benutzer. Eine Weitergabe der persönlichen Zugangsdaten/Berechtigungen an weitere Personen ist nicht erlaubt. Ä ndern sich die Aufgaben eines Mitarbeiters, hat d ie Arbeitgeber in jeweils anhand des Rechte- und Rollenkonzepts de s betreffenden Systems zu prüfen, ob die einem Mitarbeiter erteilten Berechtigungen und Zugriffsrechte zu ändern sind. Darüber hinaus hat der Arbeitgeber regelmäßig zu überprüfen, ob die erteilten Zugriffsrechte dem in der jeweiligen Konzernbetriebsvereinbarung festgelegten Rechte- und Rollenkonzept entsprechen; andernfalls hat er diese zu entziehen. Aufträge an Dritte, auch verbundene Unternehmen, dürfen dieser Vereinbarung nicht widersprechen. Dritte sind insbesondere externe Dienstleister sowie anderer Gesellschaften des Telefónica Konzerns, die nicht vom Geltungsbereich dieser KBV erfasst sind. Vertr ä ge mit Dritten sind so zu gestalten, dass die Kontrollrechte des KBR auch gegenüber Dritten wahrgenommen werden können.

======================================================================
Chunk 20 (ID: d04eb82a-b282-5754-9d90-bbe275d3b5f8) - Length: 377
Contextual Prefix: Teil dieses Dokuments ist der Abschnitt Datenschutz innerhalb der IT-Rahmenvereinbarung KBV 2.0. Hier wird klargestellt, dass die vorliegende KBV bzw. die zugeh
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 21 von 24
Datenschutz Durch diese KBV bzw. die Einzelkonzernbetriebsvereinbarungen werden keine Rechtsgrundlagen i.S.d . 4 Satz 1 BDSG iV.m . 88 DSGVO geschaffen. Dies gilt nicht, wenn dies in der jeweiligen Einzelkonzernbetriebsvereinbarung ausdrücklich unter Hinweis auf § 26 Abs. 88 DSGVO abweichend vereinbart ist.

======================================================================
Chunk 21 (ID: 73e95f79-c84b-55af-8bd3-eda721cf3081) - Length: 1974
Contextual Prefix: Dieser Abschnitt gehört zum Kapitel Umgang mit Verstößen oder Konflikten der KBV IT Rahmen 2.0. Er legt das Verfahren fest, wenn KBR oder Telefónica einen Verst
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 22 von 24
Umgang mit Verstößen oder Konflikten Stellen der KBR oder Telefónica einen Verstoß gegen diese KBV oder eine Einzel-Konzernbetriebsvereinbarung fest, kann eine der Betriebsparteien eine paritätisch besetzte Kommission (" ParKom ") anrufen, um über die möglichen Schritte zum Umgang mit dem Verstoß zu entscheiden und gemeinsam Maßnahmen zur Vermeidung von Wiederholungen des Verstoßes festzulegen. Die ParKom besteht aus zwei Mitgliedern des KBR sowie zwei Vertretern von Telefónica, wobei ein Vertreter aus dem Bereich Labour Relations kommt. Entscheidungen werden durch Mehrheitsbeschluss getroffen. Die ParKom muss unverzüglich, spätestens binnen 5 Tagen nach Anruf, zusammenkommen. Die Betriebspartei, die die ParKom anruft, unterbreitet der anderen drei Terminvorschläge. Ruft der KBR die ParKom an und lehnt Telefónica alle drei Terminvorschläge des KBR ab oder gibt keine Rückmeldung hierzu, ist die Sitzung der ParKom nicht erforderlich, sondern die Einigungsstelle kann sofort angerufen werden. Kommt eine Einigung der ParKom nicht zustande, so kann eine der Betriebsparteien die Einigungsstelle anrufen. Die Einigungsstelle besteht aus drei Beisitzenden auf jeder Seite. Vorsitzende/r der Einigungsstelle ist die in Anlage 7 an erster Stelle benannte Person. Sofern diese nicht zur Verfügung steht, werden die weiteren in Anlage 7 genannten Personen in der dort aufgeführten Reihenfolge angerufen. Regelungsgegenstand der Einigungsstelle ist es, geeignete Maßnahmen festzulegen, um eine Wiederholung des Verstoßes zu verhindern oder – wenn das nicht möglich ist - das Risiko der Wiederholung des Verstoßes so weit wie möglich ausschließen. Die Parteien unterwerfen sich in Bezug auf die Festlegung solcher Maßnahmen bereits jetzt dem Spruch der Einigungsstelle. Die Geltendmachung gesetzlicher Ansprüche des KBR wird durch dieses Verfahren weder ausgeschlossen noch ist es Voraussetzung hierfür.

======================================================================
Chunk 22 (ID: ac93223b-c4d1-55e2-b1f4-13fec62e24fa) - Length: 2280
Contextual Prefix: Dieser Abschnitt markiert den Inkrafttreten, die befristete Gültigkeit und die Folgen für bestehende Einzel-KBV-Verträge. Er regelt, wie die vorliegende KBV mit
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 23 von 24
Inkrafttreten , Ablösung und Kündigung Diese KBV tritt mit Unterzeichnung in Kraft. Die KBV ist bis zum 15.08.2025 befristet und endet automatisch mit Ende der Befristung, ohne dass es einer Kündigung bedarf. Stattdessen gilt ab dem 16.08.2025 die Anlage 8 dieser KBV. Sämtliche Einzel-Konzernbetriebsvereinbarungen, die auf Basis dieser KBV abgeschlossen wurden, gelten einschließlich der in Bezug genommenen Regelungen dieser KBV in vollem Umfang auch nach dem 15.08.2025 weiterhin. Mit dieser KBV wird die KBV „Grundlagen zum Einsatz von IT-Systemen“ (KBV Rahmen IT alt) vom 02.09./07.09.2020 abgelöst. Für IT-Systeme, für die der KBR zum Zeitpunkt des Inkrafttretens dieser KBV bereits beschlossen hatte, eine Verhandlungsgruppe oder einen Ausschuss mit der Verhandlung der Einführung und Anwendung des I T -Systems zu beauftragen, legen die Parteien gemeinsam fest, ob die KBV zur Regelung des betreffenden IT-Systems auf Basis der KBV Rahmen IT alt (ggf. mit Anpassungen) oder auf Basis dieser KBV stattfinden soll. Wenn sich die Parteien nicht einigen, gilt in diesen Fällen die KBV Rahmen IT alt. Betriebsvereinbarungen, die auf Basis der „KBV Rahmen IT alt“ abgeschlossen wurden, gelten einschließlich der in Bezug genommenen Regelungen der „KBV Rahmen IT alt “ unverändert fort. Die „Konzernbetriebsvereinbarung zum Umgang mit Mitarbeiterdaten“ vom 17.4.2012 („KBV UMM“) wird durch diese KBV Rahmen IT nicht abgelöst, sondern gilt für die Anwendung aller IT-Systeme, die auf der Grundlage eines Anhangs zur KBV UMM eingeführt worden sind, zunächst fort. Für Änderungen solcher IT-Systeme nach Inkrafttreten dieser KBV Rahmen IT gilt ausschließlich diese KBV und nicht mehr die KBV UMM. Bei der ersten mitbestimmungspflichtigen Änderung eines solchen IT-Systems nach Inkrafttreten dieser KBV Rahmen IT werden die Parteien den bisherigen Anhang zur KBV UMM durch eine KBV gemäß den Ziffer n 4 ff. Für IT-Systeme, die nach Inkrafttreten dieser KBV Rahmen IT eingeführt werden , und solche, die bereits eingeführt wurden, deren Einführung und Anwendung aber zum Zeitpunkt des Inkrafttretens dieser KBV Rahmen IT noch nicht zwischen dem KBR und Telefónica geregelt sind, findet die KBV UMM keine Anwendung.

======================================================================
Chunk 23 (ID: acc94cb6-3999-5177-8344-eaac6524fa2d) - Length: 768
Contextual Prefix: Dieses Fragment stellt den Schlussabschnitt der KBV IT Rahmen 2.0 dar. Es fasst die abschließenden Rechtsgrundlagen zusammen: keine mündlichen Abreden, Änderung
--------------------
KBV IT Rahmen 2.0 Final KORRIGIERT (23092024).docx | Chunk 24 von 24
Schlussbestimmung a) Mündliche Abreden bestehen nicht. Änderungen oder Ergänzungen dieser K BV, einschließlich dieses Schriftformerfordernisses, bedürfen zu ihrer Wirksamkeit der Schriftform. b) Sollten Regelungen dieser K BV ganz oder teilweise nichtig bzw. unwirksam sein oder werden, wird dadurch die Wirksamkeit dieser Vereinbarung im Übrigen nicht berührt. d) Änderungen der Anlagen können zwischen Betriebsparteien vorgenommen werden. Sie gelten nicht als Kündigung dieser KBV . München, den Rostock , den rows=1; columns=1; headers=Telefónica Deutschland Holding AG; sample:Telefónica Deutschland Holding AG rows=0; columns=1; headers=Konzernb etriebsrat der Telefónica Deutschland Holding AG

======================================================================
======================================================================
END OF REPORT
======================================================================