# AGENTS Leitfaden

Dieses Dokument beschreibt die Standards, Workflows und Erwartungen für Entwicklung und Beiträge an NOESIS 2.

## Ziele
- Einheitliche Arbeitsweise und reproduzierbare Builds
- Hohe Code-Qualität (Linting, Tests, Coverage)
- Klare Verantwortlichkeiten und transparente PRs

## Projektstruktur (Kurzüberblick)
- Backend-Apps: `core/`, `documents/`, `workflows/`, `ai_core/`, `users/`, `common/`, `theme/`
- Settings: `noesis2/settings/base.py`, `development.py`, `production.py`
- Frontend: `theme/static_src/input.css` → `theme/static/css/output.css`
- Entry Points: `manage.py`, `noesis2/asgi.py`, `noesis2/wsgi.py`

## Coding-Standards
- Python
  - Linting: `ruff`
  - Formatierung: `black`
  - Vor jedem Commit: `npm run lint` (oder `npm run lint:fix`)
  - Keine Secrets im Code/Repo; Konfiguration über `.env` via `django-environ`
- Frontend
  - Tailwind CSS v4 via PostCSS (`@tailwindcss/postcss` + `autoprefixer`)
  - Keine Legacy Tailwind-CLI verwenden

## Dependencies (pip-tools)
- Produktion: `requirements.in` → `pip-compile` → `requirements.txt`
- Entwicklung: `requirements-dev.in` → `pip-compile` → `requirements-dev.txt`
- Installation: `pip install -r requirements*.txt`
- .txt-Dateien nicht manuell bearbeiten; nur via `pip-compile` aktualisieren

## Settings & Secrets
- `.env.example` aktuell halten (alle notwendigen Variablen dokumentieren)
- Lokale `.env` nicht committen
- Standard-Profil: `noesis2.settings.development`; Production für Deployments nutzen

## Datenbank & Migrations
- PostgreSQL als Standard-DB; Zugang über `.env` (DB_NAME/USER/PASSWORD/HOST/PORT)
- Migrationen lokal generieren (`python manage.py makemigrations`) und anwenden (`migrate`)
- Custom User-Modell: `users.User`; bei Referenzen Importzyklen vermeiden

## Test-Philosophie & Strategie

Tests sind das Fundament für die Stabilität und Wartbarkeit von NOESIS 2. Jeder Beitrag muss von qualitativ hochwertigen Tests begleitet werden. Wir orientieren uns an der klassischen Testpyramide:

1.  **Unit-Tests (Die Basis):**
    * **Fokus:** Testen die kleinste logische Einheit (eine Funktion, eine Methode) in kompletter Isolation.
    * **Anforderung:** Abhängigkeiten wie Datenbanken, externe Dienste oder das Dateisystem **müssen** durch Mocks ersetzt werden. Diese Tests sind extrem schnell und bilden die Mehrheit unserer Testsuite.
    * **Beispiel:** Eine Funktion, die Daten transformiert, ohne auf ein Django-Model zuzugreifen.

2.  **Integrationstests (Die Mitte):**
    * **Fokus:** Testen das Zusammenspiel mehrerer Komponenten. Dies ist der häufigste Testtyp in unserem Django-Projekt.
    * **Anforderung:** Hier wird bewusst die Interaktion mit der Datenbank (`@pytest.mark.django_db`), dem Caching oder anderen internen Diensten geprüft.
    * **Beispiel:** Ein API-Endpunkt wird aufgerufen und es wird verifiziert, dass die korrekten Daten in der Datenbank angelegt und als Antwort zurückgegeben werden.

3.  **End-to-End (E2E) Tests (Die Spitze):**
    * **Fokus:** Simulieren einen vollständigen Benutzer-Workflow durch die Live-Anwendung (aus Browser-Sicht).
    * **Anforderung:** Diese Tests sind wertvoll, aber aufwändig. Sie werden nur für kritische Hauptpfade der Anwendung erstellt und sind derzeit noch nicht implementiert.

**Generelle Anweisungen:**
* **Testgetriebene Anweisungen:** Jeder Prompt zur Implementierung von Funktionalität enthält die explizite Anforderung, die notwendigen Unit- und/oder Integrationstests zu schreiben.
* **Struktur:** Tests leben in einem `tests`-Verzeichnis innerhalb der jeweiligen App und sind nach ihrer Funktion aufgeteilt (z.B. `test_models.py`, `test_views.py`, `test_tasks.py`).
* **Testdaten:** Für die Erstellung von Testdaten wird konsequent `factory-boy` verwendet. Für jedes Model existiert eine entsprechende Factory in einer `factories.py`-Datei.
* **Ausführung:** `pytest -q`
* **Coverage:** `pytest -q --cov=noesis2 --cov-report=term-missing`. Eine hohe Testabdeckung (Ziel > 80%) ist obligatorisch.


## Frontend-Workflow
- Entwicklung: `npm run dev` (Django + CSS-Watcher)
- Einmal-Build: `npm run build:css`

## CI-Empfehlungen
- Lint (ruff/black --check)
- Tests mit Coverage
- `pip-compile` Drift-Check (verifizieren, dass `requirements*.txt` aktuell sind)

## PR-Checkliste
- [ ] Lint grün (`npm run lint`)
- [ ] Tests grün (lokal/CI)
- [ ] Migrationen erstellt und angewendet (falls Modelle geändert)
- [ ] README/Docs aktualisiert
- [ ] `.env.example` konsistent zu neuen/entfernten Settings

### Datenmigrationen (PostgreSQL Best Practices)
- Drei-Phasen-Pattern für neue, nicht-nullbare FKs bei Bestandsdaten:
  1) `AddField(..., null=True, blank=True)`
  2) `RunPython` zum Backfill bestehender Zeilen
  3) `AlterField(..., null=False)`
- Backwards sicher gestalten: Beziehungen zuerst lösen (FK auf `NULL` setzen), dann nur migrierte Daten löschen (z. B. per Slug-Prefix filtern).
- Non-atomic in komplexen Fällen: `atomic = False` in der Migration setzen, um Postgres-Fehler wie „pending trigger events“ beim Zurückmigrieren zu vermeiden.
- Historische Modelle nutzen: In `RunPython` stets `apps.get_model(...)` verwenden, keine direkten Model-Imports.
- Effizient backfüllen: Nach Möglichkeit mit `update()`/Batching statt `save()` in Schleifen arbeiten.
- Optional fortgeschritten: Bei Bedarf Constraints temporär deferieren (PostgreSQL), z. B. `schema_editor.execute('SET CONSTRAINTS ALL DEFERRED')` innerhalb der `RunPython`-Funktion.

