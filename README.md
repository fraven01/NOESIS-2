# NOESIS 2

## Vision
KI-gestützte SaaS-Plattform zur prozessualen Unterstützung der betrieblichen Mitbestimmung nach § 87 Abs. 1 Nr. 6 BetrVG.

## Kernfunktionen (Geplant)
- Flexible, workflow-basierte Analyse von Dokumenten (z. B. Systembeschreibungen, Betriebsvereinbarungen)
- Mandantenfähigkeit zur Trennung von Daten verschiedener Parteien (Arbeitgeber, Betriebsräte, Anwälte)
- Wissensgenerierung und -abfrage durch angebundene Large Language Models (LLMs)
- Asynchrone Verarbeitung von rechenintensiven Analyse-Aufgaben

---

## Technologie-Stack
- Backend: Python 3.12+ mit Django 5.x
- Asynchrone Tasks: Celery & Redis
- Datenbank: PostgreSQL
- Frontend: Tailwind CSS v4 (PostCSS)
- Entwicklungsumgebung: Node.js, npm
- CI/CD & Testing: GitHub Actions, pytest

---

## Docker Quickstart
```bash
copy .env.example .env   # Linux/macOS: cp .env.example .env
docker-compose build
docker-compose up
```

---

## Lokales Setup (Alternative ohne Docker)

Docker Compose ist die bevorzugte Methode für ein konsistentes, schnelles Setup.
Die folgenden Schritte sind ein manueller Fallback, falls Docker nicht genutzt wird.

### Voraussetzungen
- Python 3.12+
- Node.js und npm
- PostgreSQL-Server
- Redis-Server

### Installations-Schritte
1. Repository klonen
   ```bash
   git clone https://github.com/fraven01/NOESIS-2.git
   cd NOESIS-2
   ```
2. Python-Umgebung einrichten
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Frontend-Abhängigkeiten installieren
   ```bash
   npm install
   ```
4. Datenbank einrichten
   - Leere PostgreSQL-Datenbank erstellen (z. B. `CREATE DATABASE noesis2_db;`).
   - `.env.example` nach `.env` kopieren und Zugangsdaten anpassen.
   - Migrationen ausführen:
     ```bash
     python manage.py migrate
     ```
5. Superuser anlegen
   ```bash
   python manage.py createsuperuser
   ```

### Entwicklungsserver starten
```bash
npm run dev
```

## Anwendung ausführen mit Docker

- `docker-compose up`: Startet die gesamte Anwendung im Vordergrund (Logs im Terminal).
- `docker-compose up -d`: Startet die Anwendung im Hintergrund (detached mode).
- `docker-compose down`: Stoppt und entfernt die Container (Volumes wie Datenbankdaten bleiben erhalten).
- `docker-compose exec web python manage.py <befehl>`: Führt einen `manage.py`-Befehl (z. B. `createsuperuser`) im laufenden `web`-Container aus.

---

## Konfiguration (.env)
Benötigte Variablen (siehe `.env.example`):

- SECRET_KEY: geheimer Schlüssel für Django
- DEBUG: `true`/`false`
- DB_NAME: Name der PostgreSQL-Datenbank
- DB_USER: DB-Benutzername
- DB_PASSWORD: DB-Passwort (Sonderzeichen werden unterstützt)
- DB_HOST: Host, z. B. `localhost`
- DB_PORT: Port, i. d. R. `5432`

Die Settings lesen `.env` via `django-environ`. Die Datenbank wird über eine zusammengesetzte `DATABASE_URL` konfiguriert (aus den Variablen oben), inkl. URL-Encoding für Sonderzeichen.

## Settings-Profile
Das alte `noesis2/settings.py` wurde entfernt; verwende ausschließlich das modulare Paket `noesis2/settings/`.

- Standard: `noesis2.settings.development` (in `manage.py`, `asgi.py`, `wsgi.py` vorkonfiguriert)
- Production: `noesis2.settings.production`
- Umstellung per Env-Var: `DJANGO_SETTINGS_MODULE=noesis2.settings.production`

## Frontend-Build (Tailwind v4 via PostCSS)
- Build/Watch: `npm run build:css` (wird in `npm run dev` automatisch gestartet)
- Konfiguration: `postcss.config.js` mit `@tailwindcss/postcss` und `autoprefixer`
- Eingabe/Ausgabe: `theme/static_src/input.css` → `theme/static/css/output.css`

## Testing
- Ausführen: `pytest -q`
- Mit Coverage: `pytest -q --cov=noesis2 --cov-report=term-missing`
- Pytest ist via `pytest.ini` auf `noesis2.settings.development` konfiguriert

## Linting & Formatierung
- Prüfen: `npm run lint` (ruff + black --check)
- Fixen: `npm run lint:fix` (ruff --fix + black)

## Abhängigkeitsmanagement (pip-tools)
- Produktion: `pip-compile requirements.in` → `requirements.txt`
- Entwicklung: `pip-compile requirements-dev.in` → `requirements-dev.txt`
- Installation: `pip install -r requirements*.txt`

## Troubleshooting (Windows)
- Falls `pytest`, `black`, `ruff` oder `pip-compile` nicht gefunden werden: `%APPDATA%\Python\Python313\Scripts` zum PATH hinzufügen.
- `.env` sollte UTF‑8 ohne BOM sein (bei Parsen-Fehlern Datei neu speichern).
