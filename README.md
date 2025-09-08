# NOESIS 2

## Vision
KI-gestützte SaaS-Plattform zur prozessualen Unterstützung der betrieblichen Mitbestimmung nach § 87 Abs. 1 Nr. 6 BetrVG.

## Kernfunktionen (Geplant)
* Flexible, workflow-basierte Analyse von Dokumenten (z.B. Systembeschreibungen, Betriebsvereinbarungen).
* Mandantenfähigkeit zur Trennung von Daten verschiedener Parteien (Arbeitgeber, Betriebsräte, Anwälte).
* Wissensgenerierung und -abfrage durch angebundene Large Language Models (LLMs).
* Asynchrone Verarbeitung von rechenintensiven Analyse-Aufgaben.

---

## Technologie-Stack
* **Backend:** Python 3.12+ mit Django 5.x
* **Asynchrone Tasks:** Celery & Redis
* **Datenbank:** PostgreSQL
* **Frontend:** Tailwind CSS
* **Entwicklungsumgebung:** Node.js, npm
* **CI/CD & Testing:** GitHub Actions, pytest

---

## Lokales Setup

### Voraussetzungen
* Python 3.12+
* Node.js und npm
* PostgreSQL-Server
* Redis-Server

### Installations-Schritte
1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/fraven01/NOESIS-2.git](https://github.com/fraven01/NOESIS-2.git)
    cd NOESIS-2
    ```
2.  **Python-Umgebung einrichten:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # oder .\.venv\Scripts\activate für Windows
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```
3.  **Frontend-Abhängigkeiten installieren:**
    ```bash
    npm install
    ```
4.  **Datenbank einrichten:**
    * Erstelle eine leere PostgreSQL-Datenbank (z.B. `CREATE DATABASE noesis2_db;`).
    * Kopiere `.env.example` zu `.env` und passe die Datenbank-Zugangsdaten an.
    * Führe die Datenbank-Migrationen aus:
    ```bash
    python manage.py migrate
    ```
5.  **Superuser anlegen:**
    ```bash
    python manage.py createsuperuser
    ```

### Entwicklungsserver starten
Stelle sicher, dass dein Redis-Server läuft. Starte dann alle Prozesse mit einem Befehl:
```bash
npm run dev