# User Management MVP (Aligned v2)

**Status:** Draft (implementierungsreif als Roadmap, nicht als Spezifikation)  
**Kontext:** Pre-MVP, Breaking Changes + DB-Reset sind ok (keine produktiven Daten).  
**Source of truth:** Code. Diese Datei beschreibt Ziele/Scope/Tasks und referenziert die aktuellen Code-Anker.

---

## TL;DR (Entscheidungen fuer MVP)

- **Admin-Oberflaeche:** Django-Admin ist **intern OK** (Operator). Keine Tenant-Admin UI in `theme/` im MVP.
- **API-Auth:** DRF Default = `IsAuthenticated` mit `SessionAuthentication` + `BasicAuthentication`; Ausnahmen nur fuer **Health/Ping**.
- **Tenant-Typen:** `ENTERPRISE` vs. `LAW_FIRM` wird **jetzt** eingefuehrt; Feature-Gating nur im **HTTP/Executor-Layer** (nicht in Graphen).
- **Rollen (DB-Values aendern, breaking ok):** `TENANT_ADMIN`, `LEGAL`, `WORKS_COUNCIL`, `MANAGEMENT`, `STAKEHOLDER` (LEGAL bleibt).
- **External Accounts:** Login mit echter Email; `account_type=EXTERNAL`, **immer case-scoped**, optional `expires_at`.
- **Case-Zugriff:** zentrale Primitive ist `CaseMembership` (many-to-many `case <-> user`), enforced in **UI und API**; Creator bekommt Auto-Grant.
- **BR (WORKS_COUNCIL):** Scope ist je Tenant konfigurierbar (`assigned` vs `all`). Verlauf/Historie-Regeln bleiben bewusst offen/stubbable.
- **Geparkt:** DataPackage/Onboarding (ehem. M4), SSO, Customer Admin UI, feingranulare Tool/Graph-Permissions.

---

## Code-Realitaet (Anker)

- Multi-Tenancy via `django-tenants`: `noesis2/settings/base.py` (`SHARED_APPS`, `TENANT_APPS`)
- Tenant-Modell: `customers/models.py`
- User+Invitation: `users/models.py`, `users/views.py` (`accept_invitation` ist `@login_required`)
- **Login-URLs fehlen aktuell:** `noesis2/urls.py` inkludiert noch nicht `django.contrib.auth.urls`
- DRF Default Auth ist aktuell leer: `noesis2/settings/base.py` (`REST_FRAMEWORK["DEFAULT_AUTHENTICATION_CLASSES"] = []`)
- Cases API ist tenant-scoped, aber nicht user-authz-scoped: `cases/api.py`
- Role-Check existiert bereits punktuell: `ai_core/authz/visibility.py` (ADMIN-only Guard)

---

## MVP Scope (was wir wirklich bauen)

### 1) Authentication Baseline (Blocker)

**Ziel:** Alles ist per Default authenticated, ausser explizit Health/Ping.

- `noesis2/settings/base.py`: DRF Defaults setzen:
  - `DEFAULT_AUTHENTICATION_CLASSES = [SessionAuthentication, BasicAuthentication]`
  - `DEFAULT_PERMISSION_CLASSES = [IsAuthenticated]`
- `noesis2/urls.py`: Django Auth URLs aktivieren (`django.contrib.auth.urls`), damit Login/Logout existieren.
- Health/Ping Endpoints explizit `AllowAny` (z.B. `api/health/*`).
- AI-Core Views: vorhandene per-view Overrides (`authentication_classes = []`) entfernen oder auf Whitelist reduzieren.

**Akzeptanzkriterien**
- [ ] `GET /cases/` ohne Auth ergibt `401/403` (nicht mehr 200).
- [ ] `GET /api/health/document-lifecycle/` ohne Auth bleibt erreichbar.
- [ ] Login-Flow ist erreichbar (z.B. `/accounts/login/`) und funktioniert tenant-scoped.

---

### 2) Tenant-Typen + HTTP-Gating

**Ziel:** Tenant kann `ENTERPRISE` oder `LAW_FIRM` sein; bestimmte Oberflaechen/Flows lassen sich pro Typ aktivieren.

- `customers/models.py`: `tenant_type` hinzufuegen (Choices `ENTERPRISE|LAW_FIRM`) + Migration.
- `customers/management/commands/create_tenant.py`: `--tenant-type` unterstuetzen (Default: `ENTERPRISE`), damit Dev-Setup sauber ist.
- Gating-Mechanik als HTTP-Layer Guard (Decorator/Permission/Middleware):
  - Beispiel: "Diese View/Route nur fuer LAW_FIRM"
  - Keine Graph-/Tool-Contract Aenderungen (siehe `AGENTS.md` Stop-Conditions).

**Akzeptanzkriterien**
- [ ] Tenant hat ein `tenant_type` Feld; bestehende Tenants werden sinnvoll gemigriert (Default ok).
- [ ] Pro Tenant kann ein LAW_FIRM Tenant "assigned-only" als Default nutzen (siehe Case Scope unten).

---

### 3) Rollen-Alignment (DB Values aendern) + Account Types

**Ziel:** Rollen-Keys spiegeln die grobe Domaene; echte Zuordnung laeuft ueber Case-Membership.

**Neue Rollen (DB Values):**
- `TENANT_ADMIN` (Operator/Admin im Tenant)
- `LEGAL` (HR/Labour-Law/Inhouse Counsel/Case-Team - bewusst grob)
- `WORKS_COUNCIL` (BR)
- `MANAGEMENT` (Fuehrung)
- `STAKEHOLDER` (Fachbereich/Projekt/Tech/Ops Stakeholder - generisch)

**Migration Mapping (alt -> neu):**
- `ADMIN -> TENANT_ADMIN`
- `LEGAL -> LEGAL`
- `BR -> WORKS_COUNCIL`
- `MANAGER -> MANAGEMENT`
- `GUEST -> STAKEHOLDER`

**External Accounts:**
- `account_type = INTERNAL|EXTERNAL`
- Optional: `expires_at` (EXTERNAL Accounts koennen einfach "ablaufen")
- Enforcement-Regel: `EXTERNAL` ist **immer** case-scoped (auch wenn Rolle "zu hoch" gesetzt waere).

**Akzeptanzkriterien**
- [ ] `profiles/models.py` enthaelt die neuen Role-Keys; DB Migration mappt alte Werte.
- [ ] Tests/Factories/Docs, die `UserProfile.Roles.*` referenzieren, sind angepasst (z.B. `users/tests/*`, `ai_core/tests/*`).
- [ ] EXTERNAL Accounts koennen sich authentifizieren, sind aber ohne CaseMembership effektiv blind.

---

### 4) CaseMembership als zentrale Permission Primitive (UI + API)

**Ziel:** "Wer sieht welchen Case?" ist eindeutig und ueberall enforced.

**Neues Modell (Vorschlag):** `cases.models.CaseMembership`
- `case` FK, `user` FK
- optional: `created_at`, `granted_by` (Operator) - MVP minimal moeglich
- Uniqueness `(case, user)`
- Indizes fuer schnelle "meine Cases" Queries

**Default Scope-Logik (MVP)**

**ENTERPRISE:**
- `TENANT_ADMIN`: all-cases
- `LEGAL`: all-cases read (write spaeter ggf. per Feature)
- `MANAGEMENT`: all-cases read (UI: wenig personenbezogene Views)
- `WORKS_COUNCIL`: per Tenant Policy (`assigned` oder `all`)
- `STAKEHOLDER`: assigned only (`CaseMembership`)
- `account_type=EXTERNAL`: assigned only (immer)

**LAW_FIRM (Default):**
- `TENANT_ADMIN`: all-cases
- alle anderen (inkl. `LEGAL`, `MANAGEMENT`): assigned only (CaseMembership)
- Optionaler spaeterer Switch per Policy ist moeglich, ohne Modell-Neuentwurf.

**Enforcement Touchpoints**
- `cases/api.py`: `get_queryset()` und `retrieve()` via "accessible cases" Query (nicht nur tenant filter).
- AI-Core HTTP Layer: alle Endpoints, die `X-Case-ID` nutzen, muessen CaseMembership pruefen, bevor Graph/Service laeuft.
- `theme/` (UI): Case-Liste/Navigation nur aus "accessible cases"; Case-Detail guarded.

**Akzeptanzkriterien**
- [ ] Nutzer ohne Membership kann fremde Cases weder via API noch UI sehen.
- [ ] `Case`-Create auto-grant: Creator erhaelt Membership (mind. in `cases/api.py`).
- [ ] EXTERNAL User kann nur Cases mit Membership sehen, unabhaengig von Rolle.

---

### 5) Operator Flows (Django-Admin) + Dev Bootstrap

**Ziel:** MVP funktioniert ohne neue Admin-UI.

- Django-Admin:
  - User anlegen (email/password)
  - `UserProfile` setzen (role, account_type, expires_at, is_active)
  - `CaseMembership` pflegen
- Bootstrap:
  - `customers/management/commands/create_tenant_superuser.py` sollte nach dem User-Create auch `UserProfile` sicherstellen (Role=`TENANT_ADMIN`, active).
  - Dev Stack (`docker-compose.dev.yml` bootstrap) muss garantieren: "ich kann mich einloggen".

**Akzeptanzkriterien**
- [ ] `npm run dev:init` / Dev-Bootstrap ergibt einen login-faehigen Tenant Admin.
- [ ] Ein Operator kann EXTERNAL Accounts + Memberships ohne Code-UI anlegen.

---

## Geparkt / Post-MVP (Backlog-Kandidaten)

- BR-Strukturen: Gremien/Ausschuesse/Verhandlungsgruppen (many-to-many; Rollen/Scopes je Gruppe)
- BR-Verlauf/Historie-Regeln (z.B. "nur aktueller Status, kein Verlauf") - bewusst offen lassen
- Manager "metrics-only" (eigene Queries/Endpoints) statt all-cases read
- Einladungsfluss ohne bestehenden Login (Invitation erzeugt User + set-password + Email Versand)
- Customer-facing Admin UI in `theme/`
- SSO/SCIM, feinere Permission Matrix, Audit Trails/Events

