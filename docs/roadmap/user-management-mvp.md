# User Management & Multi-Tenancy MVP Roadmap

**Version:** 1.1 (MVP-Sharpened)
**Erstellt:** 2025-01-19
**Letzte Änderung:** 2025-01-19 (MVP-Schärfung)
**Status:** Planning
**Ziel-Release:** MVP Q1 2025

---

## Executive Summary

Diese Roadmap definiert die **minimale** Implementierung des User-Management-Systems für NOESIS 2 mit tenant-scoped Authentifizierung, Shadow-User-System für externe Kollaboration und Data-Package-System für GDPR-konforme Cross-Tenant-Datentransfers.

**MVP-Fokus:** Funktionskern ohne Schnickschnack. Lifecycle-Management, umfangreiche UIs und harte Billing-Enforcement kommen in Phase 2.

### Zielbild (Strategic Decision)

1. **Tenant ist primäre Sicherheits- und Produktgrenze**
   - 100% Schema-Isolation (PostgreSQL-Level)
   - Keine geteilten fachlichen Datenmodelle über Tenant-Grenzen
   - Cross-Tenant-Kollaboration nur über explizite Datenpakete

2. **Zwei gleichwertige Tenant-Typen**
   - **Unternehmens-Tenant:** Mitbestimmung, Cases, interne Workflows
   - **Kanzlei-Tenant:** Mandantenverwaltung, Templates, Vorbereitung

3. **Tenant-scoped Users und Login**
   - Nutzerkonten existieren innerhalb eines Tenants
   - Kein globaler User-Login im MVP
   - Authentifizierung bleibt einfach und sicher

4. **Shadow Users für externe Case-Arbeit (MVP: Minimal)**
   - Externe Anwälte arbeiten als Shadow-Users im Unternehmens-Tenant
   - Case-scoped Permissions + Expiry (Kern-Feature)
   - **Keine** erweiterte Lifecycle-UI im MVP (→ Phase 2)

5. **Datenpakete für Ergebnis-Migration (MVP: Fokussiert)**
   - **Nur** Ergebnis-Migration (RESULT_EXPORT) im MVP
   - Vollständige Case-Exporte kommen Phase 2
   - Kanzleien bereiten vor, Unternehmen importieren bei Onboarding

6. **Kanzlei → Unternehmen Ownership-Shift (NEU)**
   - Case Ownership liegt initial bei Kanzlei
   - Nach Onboarding liegt Ownership beim Unternehmen
   - Migration ist bewusster, auditierter Akt
   - Historie bleibt Kanzlei-intern

### Non-Goals (MVP)

- ❌ Globaler Login über mehrere Tenants
- ❌ Tenant-Hopping ohne expliziten Kontextwechsel
- ❌ Live Cross-Tenant-Kollaboration
- ❌ Shared User-Identität
- ❌ Vollständige Case-Exporte (kommt Phase 2)
- ❌ Shadow-User-Lifecycle-UI (kommt Phase 2)
- ❌ Harte Billing-Enforcement (kommt Phase 2)

---

## Milestones

| Milestone | Ziel | Dauer | Status |
|-----------|------|-------|--------|
| [M1: Tenant-Typ-System](#milestone-1-tenant-typ-system) | Unterscheidung Unternehmens- vs. Kanzlei-Tenant | 1 Woche | 🔴 TODO |
| [M2: Email-as-Username](#milestone-2-email-as-username) | Vereinfachte Authentifizierung | 3 Tage | 🔴 TODO |
| [M3: Shadow-User-System (Core)](#milestone-3-shadow-user-system-core) | Case-scoped Permissions + Expiry | 1 Woche | 🔴 TODO |
| [M4: Data-Package & Onboarding](#milestone-4-data-package--onboarding) | Ergebnis-Migration + Ownership-Shift | 1.5 Wochen | 🔴 TODO |
| [M5: Production-Ready](#milestone-5-production-ready) | Tests, Monitoring, Dokumentation | 1 Woche | 🔴 TODO |

**Gesamt:** ~5 Wochen (reduziert durch MVP-Schärfung)

---

## Milestone 1: Tenant-Typ-System

**Ziel:** Unterscheidung zwischen Unternehmens-Tenant und Kanzlei-Tenant mit Vorbereitung für tenant-spezifische Features.

**Dauer:** 1 Woche
**Abhängigkeiten:** Keine
**Priorität:** 🔴 CRITICAL

### Epic 1.1: Tenant-Typ-Datenmodell

#### Scope

- Erweiterung des `Tenant`-Modells um `tenant_type`-Feld
- Enum: `ENTERPRISE` (Unternehmen) vs. `LAW_FIRM` (Kanzlei)
- Kanzlei-spezifische Konfigurationsfelder (Vorbereitung):
  - `client_count_limit` (für späteres Pricing)
  - `template_library_enabled` (Feature-Flag)
- Helper-Methoden: `is_law_firm()`, `is_enterprise()`

#### Out of Scope

- Automatische Tenant-Typ-Erkennung
- Tenant-Typ-Wechsel (wird manuell per Admin geändert)
- Kanzlei-spezifische UI (kommt in M4)

#### Akzeptanzkriterien

- [ ] Tenant-Modell hat `tenant_type`-Feld (CharField, Choices)
- [ ] Migration setzt bestehende Tenants auf `ENTERPRISE` (Default)
- [ ] Tenant-Admin kann Typ manuell ändern
- [ ] `is_law_firm()` und `is_enterprise()` funktionieren korrekt
- [ ] Tenant-Serializer (API) gibt `tenant_type` zurück

#### Security & GDPR Controls

- **Zugriffskontrolle:** Nur Tenant-Admins dürfen Typ ändern
- **Audit-Log:** Jede Änderung des Tenant-Typs wird geloggt
- **Data-Isolation:** Tenant-Typ ändert NICHT die Schema-Isolation

#### Abhängigkeiten

- Keine

#### Tests

- Unit-Tests: Tenant-Modell-Methoden (`is_law_firm()`, etc.)
- Integration-Tests: Migration setzt Default korrekt
- API-Tests: Tenant-Detail-Endpoint gibt Typ zurück

#### Audit Events

```python
AuditEvent(
    event_type="tenant.type_changed",
    actor=request.user,
    tenant=tenant,
    details={
        "old_type": "enterprise",
        "new_type": "law_firm",
        "reason": "Manual admin change"
    }
)
```

---

### Epic 1.2: Tenant-Typ-basierte Feature-Flags (Vorbereitung)

#### Scope

- **Minimales** Feature-Flag-System für tenant-spezifische Features
- Feature-Flags als **Konfiguration**, nicht als harte technische Durchsetzung
- Vorbereitung für:
  - Kanzlei-Features: Template-Bibliothek, Mandantenverwaltung, Export
  - Unternehmens-Features: Case-Lifecycle, Shadow-User-Einladungen, Import

**MVP-Schärfung:** Nur Vorbereitung (Felder im Modell, keine UI-basierte Enforcement)

#### Out of Scope

- UI-basierte Feature-Flag-Enforcement (kommt Phase 2)
- Dynamische Feature-Flags pro Tenant (alle Flags sind typ-basiert)
- Feature-Flag-Admin-UI (wird direkt in DB/Admin geändert)

#### Akzeptanzkriterien

- [ ] `FeatureFlag`-Helper-Funktion: `is_feature_enabled(tenant, feature_name)`
- [ ] Dokumentation listet geplante Features pro Tenant-Typ
- [ ] API gibt Feature-Flags im Tenant-Detail zurück (für Frontend-Logic)

#### Security & GDPR Controls

- **No Hard Enforcement:** Feature-Flags sind nur Hinweise, keine Blocker (MVP)
- **Documentation:** Dokumentiert, welche Features für welchen Typ gedacht sind

#### Abhängigkeiten

- Epic 1.1 (Tenant-Typ-Datenmodell)

#### Tests

- Unit-Tests: `is_feature_enabled(tenant, "template_library")`

#### Audit Events

- (Keine Audit-Events im MVP, nur Vorbereitung)

---

### Epic 1.3: Billing-Konfiguration (Vorbereitung, KEINE Enforcement)

#### Scope

- **Nur Vorbereitung:** Billing-Modell-Felder, KEINE harte technische Durchsetzung
- Felder im `Tenant`-Modell:
  - `billing_plan` (z.B. "starter", "professional", "enterprise")
  - `max_users` (geplantes Limit)
  - `max_shadow_users` (geplantes Limit)
  - `max_cases_per_month` (geplantes Limit)
- **Keine** Quota-Enforcement im MVP (kommt Phase 2)

**MVP-Schärfung:** Nur Datenmodell-Vorbereitung für späteres Pricing. Keine Checks, keine Blockaden.

#### Out of Scope

- Automatische Billing-Berechnung
- Payment-Gateway-Integration
- Quota-Enforcement (User kann nicht mehr Ressourcen anlegen als Limit)
- Billing-UI

#### Akzeptanzkriterien

- [ ] Tenant-Modell hat `billing_plan`, `max_users`, etc. (nullable)
- [ ] Admin-Interface zeigt Billing-Felder (read-only für Tenant-Admins)
- [ ] Dokumentation beschreibt geplantes Pricing-Model

#### Security & GDPR Controls

- **No Enforcement:** Limits sind nur Metadaten (MVP)
- **Audit-Trail:** Änderungen an Billing-Plan werden geloggt

#### Abhängigkeiten

- Epic 1.1 (Tenant-Typ-Datenmodell)

#### Tests

- Migration-Tests: Felder werden korrekt hinzugefügt
- Admin-Tests: Billing-Felder sind sichtbar

#### Audit Events

```python
AuditEvent(
    event_type="billing.plan_changed",
    actor=request.user,
    tenant=tenant,
    details={
        "old_plan": "starter",
        "new_plan": "professional",
        "reason": "Upgrade"
    }
)
```

---

## Milestone 2: Email-as-Username

**Ziel:** Vereinfachung der Authentifizierung durch Email als primären Login-Identifier.

**Dauer:** 3 Tage
**Abhängigkeiten:** Keine
**Priorität:** 🟡 MEDIUM

### Epic 2.1: User-Modell-Refactoring

#### Scope

- Entfernen des `username`-Feldes aus User-Modell
- Email als `USERNAME_FIELD` setzen
- Email-Uniqueness innerhalb des Tenant-Schemas erzwingen
- Custom `UserManager` für Email-basierte User-Erstellung
- Superuser-Creation via Email

#### Out of Scope

- Email-Verification (kommt Post-MVP)
- Email-Change-Flow (kommt Post-MVP)
- Password-Reset via Email (Django-Standard reicht)

#### Akzeptanzkriterien

- [ ] User-Modell hat KEIN `username`-Feld mehr
- [ ] `USERNAME_FIELD = "email"`
- [ ] Email ist unique im Tenant-Schema (DB-Constraint)
- [ ] `User.objects.create_user(email, password)` funktioniert
- [ ] `User.objects.create_superuser(email, password)` funktioniert
- [ ] Login-Form akzeptiert Email statt Username

#### Security & GDPR Controls

- **Email-Uniqueness:** Verhindert Duplicate-Accounts im Tenant
- **Case-Insensitive:** Email-Normalisierung (`email.lower()`) vor Speicherung
- **No PII-Leak:** Email wird in Logs maskiert (PII-Middleware)

#### Abhängigkeiten

- Keine (kann parallel zu M1 laufen)

#### Tests

- Unit-Tests: `UserManager.create_user()` mit Email
- Integration-Tests: Login mit Email funktioniert
- Migration-Tests: Bestehende User behalten Email (kein Data-Loss)

#### Audit Events

```python
AuditEvent(
    event_type="auth.login",
    actor=user,
    tenant=tenant,
    details={
        "login_method": "email_password",
        "email": user.email  # Wird von PII-Middleware maskiert
    }
)
```

---

### Epic 2.2: Invitation-System-Anpassung

#### Scope

- Invitation-Modell nutzt Email als primären Identifier
- Invitation-Acceptance erstellt User mit Email (ohne Username)
- Email-Collision-Check: Warnung, wenn Email bereits im Tenant existiert
- Multi-Invitation-Prevention: User kann nicht zweimal eingeladen werden

#### Out of Scope

- Cross-Tenant-Invitation-Check (User kann in mehreren Tenants mit gleicher Email existieren)

#### Akzeptanzkriterien

- [ ] `Invitation.email` ist indexiert
- [ ] `accept_invitation()`-View erstellt User ohne Username
- [ ] Duplicate-Email-Check vor User-Erstellung
- [ ] Fehlermeldung: "Email bereits registriert" (wenn Duplikat)

#### Security & GDPR Controls

- **Duplicate-Prevention:** Kein versehentlicher Account-Override
- **Token-Security:** Invitation-Token ist kryptographisch sicher (48 Bytes)

#### Abhängigkeiten

- Epic 2.1 (User-Modell-Refactoring)

#### Tests

- Unit-Tests: Invitation-Acceptance mit Email
- Integration-Tests: Duplicate-Email-Handling

#### Audit Events

```python
AuditEvent(
    event_type="invitation.accepted",
    actor=new_user,
    tenant=tenant,
    details={
        "invitation_id": invitation.id,
        "email": invitation.email,
        "role": invitation.role
    }
)
```

---

## Milestone 3: Shadow-User-System (Core)

**Ziel:** Minimales Shadow-User-System mit case-scoped Permissions und Expiry. **Keine** erweiterte Lifecycle-UI oder Sonderlogik im MVP.

**Dauer:** 1 Woche (reduziert durch Schärfung)
**Abhängigkeiten:** M2 (Email-as-Username)
**Priorität:** 🔴 CRITICAL

### Epic 3.1: Shadow-User-Datenmodell

#### Scope

- Erweiterung des `User`-Modells um Shadow-User-Felder:
  - `is_shadow_user` (Boolean)
  - `actual_email` (EmailField) - Echte Email des Externen
  - `access_expires_at` (DateTimeField) - Automatischer Ablauf
- Shadow-User-Identifier: `{email}.extern@{domain}` (z.B. `anna.mueller.extern@acme.de`)
- `UserProfile` bleibt gleich (Shadow-Users haben auch fachliche Rollen)

#### Out of Scope

- Automatische Shadow-User-Deaktivierung (manuell oder Cronjob kommt Phase 2)
- Shadow-User-Dashboard (kommt Phase 2)

#### Akzeptanzkriterien

- [ ] User-Modell hat `is_shadow_user`, `actual_email`, `access_expires_at`
- [ ] Shadow-User-Creation erstellt User mit `.extern@`-Email
- [ ] Migration fügt Felder hinzu (Default: `is_shadow_user=False`)
- [ ] `User.objects.filter(is_shadow_user=True)` funktioniert

#### Security & GDPR Controls

- **Expiry-Field:** Shadow-Users haben Ablaufdatum (wird im UI angezeigt)
- **Audit-Trail:** Jede Shadow-User-Creation wird geloggt mit Zweck
- **No Escalation:** Shadow-Users können nicht zu regulären Users promoted werden

#### Abhängigkeiten

- Epic 2.1 (Email-as-Username)

#### Tests

- Unit-Tests: Shadow-User-Creation
- Integration-Tests: Expiry-Date wird korrekt gesetzt

#### Audit Events

```python
AuditEvent(
    event_type="shadow_user.created",
    actor=inviting_user,
    tenant=tenant,
    details={
        "shadow_user_id": shadow_user.id,
        "actual_email": shadow_user.actual_email,
        "expires_at": shadow_user.access_expires_at.isoformat(),
        "purpose": "Legal consultation for Case #123"
    }
)
```

---

### Epic 3.2: Case-Scoped Permissions (CaseCollaborator)

#### Scope

- Neues Modell: `CaseCollaborator`
  - FK zu `Case` und `User` (nur Shadow-Users)
  - Role: `VIEWER`, `EDITOR`, `LEAD`
  - `granted_by`, `granted_at`, `expires_at`
  - `purpose` (GDPR-Zweckbindung)
- Permission-Service: `user_can_access_case(user, case)`
  - Interne Users: Zugriff auf alle Cases im Tenant
  - Shadow-Users: Nur Cases mit CaseCollaborator-Eintrag
- **Minimale** Middleware: `CaseAccessMiddleware` prüft Zugriff vor Case-Views

#### Out of Scope

- Object-Level-Permissions für andere Modelle (nur Cases)
- Granulare Permissions (z.B. "darf Kommentieren, aber nicht Editieren") - kommt Phase 2

#### Akzeptanzkriterien

- [ ] `CaseCollaborator`-Modell existiert mit allen Feldern
- [ ] `user_can_access_case()` funktioniert korrekt für beide User-Typen
- [ ] API-Endpoints werfen 403, wenn Shadow-User auf nicht-granted Case zugreift
- [ ] Case-List-API filtert Cases nach Zugriffsrechten

#### Security & GDPR Controls

- **Case-Scoped Access:** Shadow-Users sehen KEINE anderen Cases (DB-Level-Filter)
- **Purpose-Binding:** Jeder Grant hat DSGVO-Zweck (Art. 6 Abs. 1 lit. b)
- **Expiry-Enforcement:** Abgelaufene Grants werden ignoriert (DB-Query-Filter)
- **Audit-Trail:** Jeder Case-Zugriff durch Shadow-User wird geloggt

#### Abhängigkeiten

- Epic 3.1 (Shadow-User-Datenmodell)

#### Tests

- Unit-Tests: `user_can_access_case()` für alle Kombinationen
- Integration-Tests: Shadow-User kann nur granted Cases sehen
- E2E-Tests: 403-Fehler beim Zugriff auf non-granted Case

#### Audit Events

```python
AuditEvent(
    event_type="case.access_granted",
    actor=granting_user,
    tenant=tenant,
    details={
        "case_id": case.id,
        "shadow_user_id": shadow_user.id,
        "role": "VIEWER",
        "expires_at": expires_at.isoformat(),
        "purpose": "Legal review"
    }
)

AuditEvent(
    event_type="case.access_denied",
    actor=shadow_user,
    tenant=tenant,
    details={
        "case_id": case.id,
        "reason": "No CaseCollaborator grant"
    }
)
```

---

### Epic 3.3: Shadow-User-Invitation (Minimal)

#### Scope

- **Minimaler** Service: `invite_external_collaborator(case, email, role, purpose, expires_in_days)`
  - Erstellt Shadow-User (oder holt bestehenden)
  - Erstellt `CaseCollaborator`-Grant
  - **Keine** Email-Versand im MVP (manuell oder kommt Phase 2)
- **Keine** UI im MVP (Admin-Interface reicht)
  - Admin kann Shadow-User über Django-Admin anlegen
  - Admin kann CaseCollaborator über Django-Admin verknüpfen

**MVP-Schärfung:** Nur Kern-Service, keine UI, keine Email-Automation.

#### Out of Scope

- UI-Button "Externen einladen" (kommt Phase 2)
- Email-Template & Versand (kommt Phase 2)
- Passwort-Setup-Flow (Admin setzt Passwort manuell)
- SSO-Integration

#### Akzeptanzkriterien

- [ ] Service `invite_external_collaborator()` existiert
- [ ] Service erstellt Shadow-User + CaseCollaborator korrekt
- [ ] Admin kann Shadow-User über Django-Admin anlegen
- [ ] Admin kann CaseCollaborator über Django-Admin verknüpfen

#### Security & GDPR Controls

- **Invitation-Only:** Shadow-Users können sich NICHT selbst registrieren
- **Purpose-Requirement:** Service verlangt Zweck (DSGVO)
- **Expiry-Default:** Default 90 Tage (kann angepasst werden)

#### Abhängigkeiten

- Epic 3.1 (Shadow-User-Datenmodell)
- Epic 3.2 (CaseCollaborator)

#### Tests

- Unit-Tests: `invite_external_collaborator()` erstellt korrekte Objekte
- Integration-Tests: Shadow-User-Login funktioniert nach Creation

#### Audit Events

```python
AuditEvent(
    event_type="shadow_user.invited",
    actor=inviting_user,
    tenant=tenant,
    details={
        "case_id": case.id,
        "shadow_user_email": shadow_user.email,
        "actual_email": actual_email,
        "role": "LEGAL",
        "case_role": "VIEWER",
        "purpose": purpose,
        "expires_at": expires_at.isoformat()
    }
)
```

---

## Milestone 4: Data-Package & Onboarding

**Ziel:** Minimales Data-Package-System für **Ergebnis-Migration** + Kanzlei→Unternehmen Onboarding-Flow.

**Dauer:** 1.5 Wochen
**Abhängigkeiten:** M1 (Tenant-Typ-System)
**Priorität:** 🟠 HIGH

### Epic 4.1: DataPackage-Datenmodell (Minimal)

#### Scope

- Neues Modell: `DataPackage`
  - `package_type`: **Nur** `RESULT_EXPORT` im MVP (keine CASE_EXPORT, TEMPLATE_EXPORT)
  - `status`: `CREATED`, `SEALED`, `IMPORTED`, `EXPIRED`, `REVOKED`
  - `source_tenant_name` (String, kein FK!)
  - `target_tenant_name` (Optional)
  - `exported_by`, `exported_at`
  - `imported_at` (nullable)
  - `purpose` (GDPR-Zweckbindung)
  - `expires_at` (Ablaufdatum)
  - `signature` (HMAC-SHA256)
  - `payload` (JSONField, encrypted)
  - `payload_hash` (SHA256)
  - `access_log` (JSONField, Audit-Trail)
- Payload-Struktur (MVP):
  - **Result-Export:** Nur Endergebnis (z.B. BV-Entwurf, Gutachten)
  - **Keine** vollständigen Case-Exports (kommt Phase 2)

**MVP-Schärfung:** Nur RESULT_EXPORT (Endergebnisse), keine kompletten Cases.

#### Out of Scope

- CASE_EXPORT (vollständiger Case mit allen Dokumenten) - Phase 2
- TEMPLATE_EXPORT - Phase 2
- Binary-Payload (nur JSON)
- Compression

#### Akzeptanzkriterien

- [ ] `DataPackage`-Modell existiert mit Feldern
- [ ] `package_type` erlaubt nur `RESULT_EXPORT` (Enum)
- [ ] `payload` ist JSONField
- [ ] Migration erstellt Tabelle

#### Security & GDPR Controls

- **No Cross-Schema-FK:** `source_tenant_name` ist String (keine DB-Referenz)
- **Signature:** HMAC-SHA256 verhindert Manipulation
- **Expiry:** Pakete werden nach Ablauf ungültig
- **Purpose-Binding:** Jedes Paket hat DSGVO-Zweck

#### Abhängigkeiten

- Keine

#### Tests

- Unit-Tests: DataPackage-Creation
- Migration-Tests: Tabelle korrekt erstellt

#### Audit Events

```python
AuditEvent(
    event_type="data_package.created",
    actor=exporting_user,
    tenant=source_tenant,
    details={
        "package_id": package.id,
        "package_type": "RESULT_EXPORT",
        "target_tenant": target_tenant_name,
        "purpose": purpose,
        "expires_at": expires_at.isoformat()
    }
)
```

---

### Epic 4.2: Result-Export-Service (Kanzlei → Paket)

#### Scope

- Service: `export_result_package(case, result_data, purpose, target_tenant=None, expires_in_days=30)`
  - Serialisiert **nur Endergebnis** (z.B. BV-Entwurf als JSON/PDF-Base64)
  - Erstellt `DataPackage` mit minimalem Payload
  - Berechnet `payload_hash` (SHA256)
  - Signiert mit `TENANT_DATA_PACKAGE_SECRET` (HMAC)
  - Setzt Status auf `SEALED`
- Payload-Struktur (MVP):
  ```json
  {
    "result_type": "bv_entwurf",  // oder "gutachten", "stellungnahme"
    "case_reference": {
      "title": "Case Title",
      "case_id": "123",  // Kanzlei-interne ID (nicht übertragbar)
      "client_name": "ACME Corp"
    },
    "result": {
      "content": "...",  // Markdown oder HTML
      "attachments": [
        {"filename": "entwurf.pdf", "content_base64": "..."}
      ]
    }
  }
  ```
- Encryption: Payload wird mit `TENANT_DATA_PACKAGE_SECRET` verschlüsselt (Fernet)

**MVP-Schärfung:** Nur Endergebnisse (Outputs), keine Case-Metadaten, keine Related Objects.

#### Out of Scope

- Vollständiger Case-Export (kommt Phase 2)
- Multi-Case-Export

#### Akzeptanzkriterien

- [ ] `export_result_package()` erstellt Paket mit minimalem Payload
- [ ] Payload enthält nur Endergebnis (kein Case)
- [ ] Signature ist korrekt
- [ ] Status ist `SEALED` nach Export
- [ ] Payload ist encrypted

#### Security & GDPR Controls

- **Encryption:** Payload ist encrypted (Fernet)
- **Signature:** HMAC-SHA256 verhindert Manipulation
- **Purpose-Required:** Export ohne Zweck wird abgelehnt
- **No PII-Leak:** Payload wird NICHT in Logs ausgegeben

#### Abhängigkeiten

- Epic 4.1 (DataPackage-Datenmodell)

#### Tests

- Unit-Tests: `export_result_package()` serialisiert korrekt
- Security-Tests: Signature-Verify funktioniert

#### Audit Events

```python
AuditEvent(
    event_type="data_package.sealed",
    actor=exporting_user,
    tenant=source_tenant,
    details={
        "package_id": package.id,
        "case_id": case.id,
        "result_type": "bv_entwurf",
        "payload_size_bytes": len(json.dumps(payload))
    }
)
```

---

### Epic 4.3: Result-Import-Service (Paket → Unternehmen)

#### Scope

- Service: `import_result_package(package_id, importing_user)`
  - Lädt `DataPackage` aus DB
  - Prüft Status (muss `SEALED` sein)
  - Prüft Expiry
  - Verifiziert Signature (HMAC-Check)
  - Decrypted Payload
  - Erstellt **neuen Case** im Unternehmens-Tenant mit importiertem Ergebnis
  - Setzt Package-Status auf `IMPORTED`
- Deserialisierung:
  - Ergebnis → Neuer Case (mit Status "imported" oder "onboarded")
  - Attachments → Neue Document-Objekte
  - **Keine** Historie-Übernahme (Case-ID aus Kanzlei ist nur Referenz)

**MVP-Schärfung:** Import erstellt neuen Case, keine Merge-Logic.

#### Out of Scope

- Selective-Import
- Merge-Logic (wenn Case bereits existiert)

#### Akzeptanzkriterien

- [ ] `import_result_package()` erstellt Case im Ziel-Tenant
- [ ] Ergebnis wird als Case-Content importiert
- [ ] Package-Status wird auf `IMPORTED` gesetzt
- [ ] Audit-Log enthält Import-Details

#### Security & GDPR Controls

- **Signature-Verify:** Import wird abgelehnt, wenn Signatur ungültig
- **Expiry-Check:** Abgelaufene Pakete können nicht importiert werden
- **Audit-Trail:** Import wird geloggt
- **No Overwrite:** Import erstellt IMMER neuen Case

#### Abhängigkeiten

- Epic 4.1 (DataPackage-Datenmodell)
- Epic 4.2 (Export-Service)

#### Tests

- Unit-Tests: `import_result_package()` deserialisiert korrekt
- Integration-Tests: Import erstellt Case im Ziel-Tenant
- E2E-Tests: Export → Import funktioniert

#### Audit Events

```python
AuditEvent(
    event_type="data_package.imported",
    actor=importing_user,
    tenant=target_tenant,
    details={
        "package_id": package.id,
        "source_tenant": package.source_tenant_name,
        "case_id": new_case.id,
        "result_type": "bv_entwurf"
    }
)
```

---

### Epic 4.4: Package-Lifecycle (Minimal)

#### Scope

- **Minimale** Lifecycle-Verwaltung:
  - Manuelles Setzen von `status` auf `EXPIRED` (Admin-Interface)
  - Manuelles `revoke_package(package_id, reason)` (Service)
- **Kein** automatischer Cronjob im MVP (kommt Phase 2)

**MVP-Schärfung:** Nur manuelle Verwaltung, kein Automation.

#### Out of Scope

- Cronjob `expire_data_packages` (Phase 2)
- Package-Liste-UI (Phase 2)
- Bulk-Actions (Phase 2)

#### Akzeptanzkriterien

- [ ] Admin kann Package-Status manuell ändern
- [ ] `revoke_package()` setzt Status auf `REVOKED` und löscht Payload
- [ ] Revoked Packages können nicht importiert werden

#### Security & GDPR Controls

- **Manual Revoke:** Admin kann Pakete widerrufen (DSGVO Art. 17)
- **Audit-Trail:** Revoke wird geloggt

#### Abhängigkeiten

- Epic 4.1 (DataPackage-Datenmodell)

#### Tests

- Unit-Tests: `revoke_package()` funktioniert
- Integration-Tests: Revoked Package kann nicht importiert werden

#### Audit Events

```python
AuditEvent(
    event_type="data_package.revoked",
    actor=revoking_user,
    tenant=tenant,
    details={
        "package_id": package.id,
        "reason": reason,
        "payload_deleted": True
    }
)
```

---

### Epic 4.5: Kanzlei → Unternehmen Onboarding-Flow (NEU)

#### Scope

- **Ownership-Shift-Workflow:** Expliziter Prozess für Kanzlei→Unternehmen-Migration
- Service: `onboard_client(client_data, kanzlei_tenant, result_packages)`
  - Erstellt neuen Unternehmens-Tenant (oder linkt zu bestehendem)
  - Importiert alle Result-Packages (aus `result_packages`-Liste)
  - Setzt Ownership der importierten Cases auf Unternehmen
  - Audit-Log: "Onboarded from Kanzlei X"
- Case-Ownership:
  - Initial: Case gehört Kanzlei (Kanzlei-Tenant)
  - Nach Onboarding: Case gehört Unternehmen (Unternehmens-Tenant)
  - Kanzlei behält Historie (kein Datenverlust)
- **Keine** Live-Synchronisation (einmaliger Transfer)

**Zielbild:** Case-Ownership wechselt explizit und auditiert von Kanzlei zu Unternehmen.

#### Out of Scope

- Automatisches Onboarding (nur manuell per Admin)
- Rück-Migration (Unternehmen → Kanzlei)
- Multi-Tenant-Ownership (ein Case gehört immer genau einem Tenant)

#### Akzeptanzkriterien

- [ ] Service `onboard_client()` erstellt Unternehmens-Tenant (oder linkt)
- [ ] Alle Result-Packages werden importiert
- [ ] Cases haben Metadaten: `onboarded_from_tenant`, `onboarded_at`
- [ ] Kanzlei-Cases bleiben im Kanzlei-Tenant (Historie)
- [ ] Audit-Log dokumentiert Ownership-Shift

#### Security & GDPR Controls

- **Explicit Transfer:** Kein automatischer Datentransfer, nur auf Anfrage
- **Audit-Trail:** Jeder Onboarding-Vorgang wird geloggt
- **No Data-Loss:** Kanzlei behält Original-Cases
- **GDPR-Compliance:** Zweck des Transfers ist dokumentiert (Vertragsdurchführung)

#### Abhängigkeiten

- Epic 4.2 (Export-Service)
- Epic 4.3 (Import-Service)

#### Tests

- Unit-Tests: `onboard_client()` erstellt korrekte Struktur
- Integration-Tests: Cases werden korrekt importiert
- E2E-Tests: Kompletter Onboarding-Flow (Kanzlei exportiert → Unternehmen onboarded)

#### Audit Events

```python
AuditEvent(
    event_type="client.onboarded",
    actor=onboarding_user,
    tenant=new_enterprise_tenant,
    details={
        "source_tenant": kanzlei_tenant.name,
        "result_packages_imported": [pkg.id for pkg in result_packages],
        "cases_created": [case.id for case in imported_cases],
        "onboarded_at": timezone.now().isoformat()
    }
)
```

---

## Milestone 5: Production-Ready

**Ziel:** Tests, Monitoring, Dokumentation und Produktionsreife.

**Dauer:** 1 Woche
**Abhängigkeiten:** M1-M4 abgeschlossen
**Priorität:** 🔴 CRITICAL

### Epic 5.1: Comprehensive Testing

#### Scope

- Unit-Tests für alle Services:
  - Shadow-User-Creation
  - CaseCollaborator-Permissions
  - DataPackage-Export/Import
  - Onboarding-Flow
- Integration-Tests für komplette Workflows:
  - Shadow-User-Invitation → Login → Case-Zugriff
  - Result-Export → Import → Case-Creation
  - Kanzlei-Onboarding → Unternehmen-Tenant-Creation
- E2E-Tests (Playwright):
  - Kanzlei exportiert Result → Unternehmen importiert
  - Admin lädt Shadow-User ein → Shadow-User loggt ein
- Security-Tests:
  - Shadow-User kann NICHT auf non-granted Cases zugreifen
  - Ungültige Package-Signatur wird abgelehnt
  - Abgelaufene Shadow-Users können nicht einloggen (manuelle Deaktivierung)

#### Out of Scope

- Load-Tests (kommt Post-MVP)
- Penetration-Tests (kommt Post-MVP)
- Performance-Tests (kommt Post-MVP)

#### Akzeptanzkriterien

- [ ] 80%+ Code-Coverage für kritische Services (100% für Security-critical)
- [ ] Alle E2E-Tests grün
- [ ] Security-Tests decken alle Zugriffskontroll-Szenarien ab

#### Security & GDPR Controls

- **Security-First:** Alle Security-Tests müssen grün sein vor Release
- **No Bypass:** Tests prüfen, dass Guards nicht umgangen werden können

#### Abhängigkeiten

- M1-M4 abgeschlossen

#### Tests

- Test-Coverage-Report: `npm run test:py:cov`
- E2E-Report: `npm run e2e`

#### Audit Events

- (Keine neuen Audit-Events, nur Testing)

---

### Epic 5.2: Monitoring & Observability (Minimal)

#### Scope

- Langfuse-Traces für kritische Workflows:
  - Shadow-User-Invitation
  - Result-Export/Import
  - Onboarding-Flow
- ELK-Logs für Audit-Events:
  - Alle `AuditEvent`-Logs gehen nach ELK
  - **Kein** Custom-Dashboard im MVP (Standard-Kibana reicht)
- **Keine** Alerts im MVP (manuelles Monitoring)

**MVP-Schärfung:** Nur Basis-Logging, keine Alerts, keine Custom-Dashboards.

#### Out of Scope

- Custom-Dashboards (Phase 2)
- Alerts (Phase 2)
- Metrics (Prometheus/Grafana kommt Phase 2)

#### Akzeptanzkriterien

- [ ] Langfuse-Traces für Onboarding-Flow vorhanden
- [ ] ELK-Logs enthalten alle Audit-Events
- [ ] Logs sind durchsuchbar in Kibana

#### Security & GDPR Controls

- **No PII in Traces:** PII-Middleware maskiert Emails
- **Audit-Log-Retention:** 90 Tage (GDPR)

#### Abhängigkeiten

- M1-M4 abgeschlossen

#### Tests

- Integration-Tests: Trace wird erstellt

#### Audit Events

- (Keine neuen Audit-Events, nur Monitoring)

---

### Epic 5.3: Dokumentation (Minimal)

#### Scope

- **User-Documentation (Minimal):**
  - Kanzlei-Guide: "Wie exportiere ich ein Ergebnis?"
  - Unternehmens-Guide: "Wie importiere ich ein Datenpaket?"
  - **Kein** Externe-Guide (Shadow-User-Verwaltung ist Admin-only)
- **Admin-Documentation:**
  - Tenant-Typ-Konfiguration
  - Shadow-User-Verwaltung (manuell über Django-Admin)
  - Onboarding-Flow
- **Developer-Documentation:**
  - `users/README.md`: Shadow-User-System (Update)
  - `cases/README.md`: CaseCollaborator-Permissions (Update)
  - `packages/README.md`: DataPackage-System (neu)
- **Runbooks (Minimal):**
  - "Shadow-User löschen" (DSGVO-Request)
  - "Data-Package widerrufen"
  - "Client onboarden"

**MVP-Schärfung:** Nur essenzielle Dokumentation, keine umfangreichen Guides.

#### Out of Scope

- Video-Tutorials
- Umfangreiche User-Guides
- API-Schema-Update (kommt Phase 2, wenn UI gebaut wird)

#### Akzeptanzkriterien

- [ ] Kanzlei-Guide und Unternehmens-Guide geschrieben
- [ ] Developer-Docs up-to-date
- [ ] Runbooks getestet (Dry-Run)

#### Security & GDPR Controls

- **DSGVO-Runbook:** Prozess für DSGVO-Anfragen dokumentiert

#### Abhängigkeiten

- M1-M4 abgeschlossen

#### Tests

- Doc-Tests: Links funktionieren
- Runbook-Tests: Dry-Run

#### Audit Events

- (Keine neuen Audit-Events, nur Dokumentation)

---

### Epic 5.4: Deployment & Migration

#### Scope

- **Migrations:**
  - Tenant-Typ-Migration: Setzt alle Tenants auf `ENTERPRISE`
  - User-Model-Migration: Fügt Shadow-User-Felder hinzu
  - DataPackage-Migration: Erstellt Tabelle
  - CaseCollaborator-Migration: Erstellt Tabelle
- **Deployment-Plan:**
  - Staging-Deployment (1 Tag Testing)
  - Production-Deployment
  - Rollback-Plan
- **Smoke-Tests:**
  - Login mit Email funktioniert
  - Shadow-User kann Case mit Grant sehen
  - Result-Export/Import funktioniert
  - Onboarding-Flow funktioniert

#### Out of Scope

- Zero-Downtime-Deployment (kurze Downtime akzeptabel)

#### Akzeptanzkriterien

- [ ] Alle Migrations laufen fehlerfrei (Staging)
- [ ] Smoke-Tests grün
- [ ] Production-Deployment erfolgreich
- [ ] Rollback-Plan dokumentiert

#### Security & GDPR Controls

- **Migration-Safety:** Migrations nur ADD-Columns (kein Data-Loss)
- **Backup:** Full-DB-Backup vor Deployment

#### Abhängigkeiten

- M1-M4 abgeschlossen
- Epic 5.1-5.3 abgeschlossen

#### Tests

- Migration-Tests: `npm run test:py:migrations`
- Smoke-Tests: Manuell in Staging

#### Audit Events

```python
AuditEvent(
    event_type="deployment.completed",
    actor="system",
    tenant=None,
    details={
        "version": "2.0.0-mvp",
        "migrations_run": ["0001_tenant_type", "0002_shadow_user", "0003_data_package", "0004_case_collaborator"],
        "deployment_time": timezone.now().isoformat()
    }
)
```

---

## Success Criteria

**MVP ist erfolgreich, wenn:**

1. ✅ **Tenant-Typ-System:**
   - Kanzlei- und Unternehmens-Tenants sind unterscheidbar
   - Billing-Felder sind vorbereitet (keine Enforcement nötig)

2. ✅ **Email-as-Username:**
   - Login funktioniert mit Email

3. ✅ **Shadow-User-System (Core):**
   - Externe können zu Cases eingeladen werden (manuell über Admin)
   - Case-Scoped-Permissions funktionieren
   - Abgelaufene Shadow-Users sind erkennbar (Expiry-Feld)

4. ✅ **Data-Package-System (Ergebnis-Migration):**
   - Kanzleien können Ergebnisse exportieren
   - Unternehmen können Pakete importieren
   - Signatur und Expiry funktionieren

5. ✅ **Onboarding-Flow:**
   - Kanzlei kann Client onboarden
   - Cases werden korrekt in Unternehmens-Tenant importiert
   - Ownership-Shift ist auditiert

6. ✅ **Production-Ready:**
   - Alle Tests grün (Unit, Integration, E2E, Security)
   - Basis-Monitoring läuft (Langfuse, ELK)
   - Dokumentation ist vollständig
   - Deployment erfolgreich

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Shadow-User-Permissions komplex | MEDIUM | HIGH | Prototyping in Epic 3.2, frühe Tests |
| Package-Signatur-Algorithmus unsicher | LOW | HIGH | Use HMAC-SHA256 (battle-tested), kein Custom-Crypto |
| Onboarding-Flow unklar | MEDIUM | MEDIUM | Detailliertes Epic 4.5, frühe User-Interviews |
| GDPR-Non-Compliance bei Expiry | LOW | HIGH | Legal-Review, Runbook-Testing |
| Migration-Failure | LOW | CRITICAL | Full-Backup, Rollback-Plan |

---

## Phase 2 (Post-MVP)

**Folgende Features kommen NACH MVP:**

- Shadow-User-Lifecycle-UI (Cronjobs, Benachrichtigungen)
- Vollständige Case-Exporte (CASE_EXPORT)
- Template-Exporte (TEMPLATE_EXPORT)
- Harte Billing-Enforcement (Quota-Checks)
- Package-Liste-UI (Export/Import-Browser)
- Automatische Package-Expiry (Cronjobs)
- Alerts & Custom-Dashboards
- Granulare Shadow-User-Permissions (z.B. "darf kommentieren, nicht editieren")
- SSO-Integration
- Email-Automation für Invitations

---

## Appendix

### Glossar

| Begriff | Definition |
|---------|------------|
| **Tenant-Scoped User** | User-Account existiert innerhalb eines Tenant-Schemas (nicht global) |
| **Shadow User** | Externer User mit case-scoped Permissions im Unternehmens-Tenant |
| **Data Package** | Verschlüsseltes, signiertes Export-Paket für Cross-Tenant-Transfers |
| **CaseCollaborator** | Grant-Objekt für Shadow-User-Zugriff auf spezifischen Case |
| **Tenant Type** | ENTERPRISE (Unternehmen) vs. LAW_FIRM (Kanzlei) |
| **Result-Export** | Export nur des Endergebnisses (z.B. BV-Entwurf), nicht des vollständigen Cases |
| **Ownership-Shift** | Wechsel der Case-Ownership von Kanzlei zu Unternehmen (Onboarding) |

### Referenzen

- [AGENTS.md](../AGENTS.md) - Tool-Verträge und Architektur
- [CLAUDE.md](../CLAUDE.md) - Entwickler-Guide
- [docs/multi-tenancy.md](../multi-tenancy.md) - Multi-Tenancy-Architektur
- [docs/security/secrets.md](../security/secrets.md) - Secret-Management
- [users/README.md](../../users/README.md) - User-Model-Dokumentation

---

**Version:** 1.1 (MVP-Sharpened)
**Zuletzt aktualisiert:** 2025-01-19
**Nächstes Review:** Nach M3 (Shadow-User-System Core)
**Freigegeben:** Ja (mit MVP-Schärfungen)
