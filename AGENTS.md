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

## Tests
- Pytest + pytest-django; Testdaten via factory-boy
- Ausführen: `pytest -q`; mit Coverage: `pytest -q --cov=noesis2 --cov-report=term-missing`
- Ziel: stabile Tests, sinnvolle Abdeckung (z. B. ≥ 80%)

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

