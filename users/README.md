# Users App

Django-App für Benutzerverwaltung und Einladungssystem in NOESIS 2. Implementiert Custom User Model und rollenbasiertes Invitation-System mit Integration in das Multi-Tenancy-Framework.

## Überblick

Die `users` App ist die zentrale Authentifizierungs- und Benutzerverwaltungskomponente:

- **Custom User Model**: Erweiterung von Django's `AbstractUser`
- **Invitation System**: Token-basierte Benutzereinladungen mit Rollen
- **Profile Integration**: Nahtlose Verknüpfung mit `profiles` App
- **Multi-Tenancy Ready**: Kompatibel mit `django-tenants` Schema-Isolation

## Architektur

```
users/
├── models.py          # User & Invitation Models
├── views.py           # Accept Invitation View
├── admin.py           # Django Admin Interface
├── migrations/        # DB Schema Migrationen
│   ├── 0001_initial.py      # User Model (2025-09-08)
│   └── 0002_invitation.py   # Invitation System (2025-09-10)
└── tests/
    ├── factories.py                 # Test Factories
    ├── test_models.py              # Model Tests
    └── test_invitation_flow.py     # Integration Tests
```

## Models

### User

**Location**: [users/models.py:8-9](users/models.py#L8-L9)

```python
class User(AbstractUser):
    pass
```

Minimale Erweiterung von Django's `AbstractUser` ohne zusätzliche Felder. Alle benutzerdefinierten Daten werden in der verknüpften `UserProfile` gespeichert (Separation of Concerns).

**Standardfelder** (von `AbstractUser` geerbt):
- `username`: Unique, 150 chars max
- `email`: EmailField (nicht unique!)
- `first_name`, `last_name`: Optional
- `password`: Hashed
- `is_staff`, `is_superuser`, `is_active`: Permissions
- `date_joined`, `last_login`: Timestamps
- `groups`, `user_permissions`: M2M Relations

**Konfiguration**:
```python
# noesis2/settings/base.py:341
AUTH_USER_MODEL = "users.User"
```

**Verwendung**:
```python
from django.contrib.auth import get_user_model

User = get_user_model()
user = User.objects.create_user(username="john", email="john@example.com", password="secret")
```

### Invitation

**Location**: [users/models.py:12-26](users/models.py#L12-L26)

```python
class Invitation(models.Model):
    email = models.EmailField()
    role = models.CharField(max_length=20, choices=UserProfile.Roles.choices)
    token = models.CharField(max_length=64, unique=True, blank=True, null=True)
    expires_at = models.DateTimeField(blank=True, null=True)
    accepted_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

Token-basiertes Einladungssystem für neue Benutzer mit automatischem Rollenzuweisung.

**Felder**:
- `email`: Ziel-E-Mail (nicht validiert gegen existierende User)
- `role`: Rolle aus `UserProfile.Roles` (`ADMIN`, `LEGAL`, `BR`, `MANAGER`, `GUEST`)
- `token`: 16-Byte URL-safe Token (generiert via `secrets.token_urlsafe`)
- `expires_at`: Ablaufdatum (default: +7 Tage)
- `accepted_at`: Timestamp des Akzeptierens (null = noch offen)
- `created_at`: Erstellungszeitpunkt

**Methoden**:

#### `generate_token()`

Generiert neuen Token und setzt Ablaufdatum.

```python
invitation = Invitation.objects.create(email="new@example.com", role="GUEST")
token = invitation.generate_token()
invitation.save()
# token ist 22 Zeichen lang (16 Bytes Base64)
# expires_at ist now() + 7 Tage
```

**Lifecycle**:
1. **Erstellen**: Admin erstellt Invitation mit E-Mail + Rolle
2. **Token generieren**: Manuell oder via Admin-Action
3. **Versenden**: E-Mail mit Link `/invite/accept/{token}/` (aktuell Stub)
4. **Akzeptieren**: User klickt Link → Login → Rolle wird zugewiesen
5. **Abschließen**: `accepted_at` wird gesetzt, Token ist verbraucht

**Validierung**:
- Token muss unique sein (DB-Constraint)
- Expired: `expires_at < now()`
- Akzeptiert: `accepted_at is not None`
- Beide Zustände führen zu 404 beim Akzeptieren

## Views

### accept_invitation

**Location**: [users/views.py:10-23](users/views.py#L10-L23)
**URL**: `/invite/accept/<token>/` ([noesis2/urls.py:54](noesis2/urls.py#L54))
**Decorator**: `@login_required`

Akzeptiert Einladung und weist Rolle zu.

**Flow**:
1. User muss eingeloggt sein (Redirect zu Login wenn nicht)
2. Token wird validiert (`get_object_or_404`)
3. Checks:
   - Bereits akzeptiert? → 404
   - Abgelaufen? → 404
4. UserProfile wird geladen/erstellt (`ensure_user_profile`)
5. Rolle wird gesetzt: `profile.role = invitation.role`
6. Profil wird aktiviert: `profile.is_active = True`
7. Invitation wird markiert: `invitation.accepted_at = now()`
8. Redirect zu `/`

**Sicherheit**:
- Login-Pflicht verhindert anonyme Nutzung
- Token ist 128-bit (sehr hohe Entropie)
- One-time use: `accepted_at` verhindert Mehrfachnutzung
- Expiry: Zeitlich begrenzte Gültigkeit

**Fehlerbehandlung**:
- Ungültiger Token → 404 (kein Information Leak)
- Expired/Accepted → 404 (identisches Verhalten)

## Admin Interface

**Location**: [users/admin.py:9-36](users/admin.py#L9-L36)

### InvitationAdmin

Django Admin-Konfiguration für Invitation-Management.

**List Display**:
```python
list_display = ("email", "role", "token", "expires_at", "accepted_at", "created_at")
```

Zeigt alle relevanten Felder in der Admin-Liste.

**Actions**:

#### `invite_guest`

Admin-Action zum Versenden von Einladungen.

**Funktion**:
1. Prüft ob bereits akzeptiert → Warning
2. Generiert Token falls nicht vorhanden
3. Setzt `expires_at` auf +7 Tage falls nicht gesetzt
4. Speichert Invitation
5. Zeigt Einladungslink in Admin-Messages
6. Print-Stub für E-Mail-Versand

**Verwendung**:
1. Admin wählt Invitations in der Liste
2. Wählt "Gast einladen" aus Actions-Dropdown
3. Erhält Link im Format `/invite/accept/{token}/`
4. Versendet Link manuell (oder via zukünftiges E-Mail-System)

**E-Mail-Integration**:
```python
# Aktuell Stub:
print(f"Stub mail to {invitation.email}: /invite/accept/{invitation.token}/")

# Zukünftig (z.B. mit django-anymail):
# send_mail(
#     subject="Einladung zu NOESIS 2",
#     message=f"Bitte akzeptieren Sie: {settings.BASE_URL}/invite/accept/{invitation.token}/",
#     from_email=settings.DEFAULT_FROM_EMAIL,
#     recipient_list=[invitation.email],
# )
```

## Integration mit anderen Apps

### profiles App

**UserProfile** wird automatisch bei User-Erstellung angelegt (Signal).

**Location**: [profiles/signals.py:9](profiles/signals.py#L9)

```python
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
```

**Invitation-Flow**:
1. User existiert bereits (Login-Pflicht)
2. `ensure_user_profile(user)` lädt oder erstellt Profile
3. Rolle aus Invitation wird auf Profile übertragen
4. Profile wird aktiviert (`is_active=True`)

**Rollen** ([profiles/models.py](profiles/models.py)):
```python
class Roles(models.TextChoices):
    ADMIN = "ADMIN", "Admin"
    LEGAL = "LEGAL", "Legal"
    BR = "BR", "BR"
    MANAGER = "MANAGER", "Manager"
    GUEST = "GUEST", "Guest"
```

### organizations App

**Organization.members**: M2M zu `settings.AUTH_USER_MODEL`

**Location**: [organizations/models.py:32](organizations/models.py#L32)

User können Mitglied mehrerer Organisationen sein (N:M).

## Testing

### Factories

**Location**: [users/tests/factories.py](users/tests/factories.py)

#### UserFactory

```python
user = UserFactory()  # username="user1", email="user1@example.com"
user = UserFactory(email="custom@example.com")
user = UserFactory(username="admin", is_staff=True)
```

**Features**:
- Auto-Sequence für `username` (`user0`, `user1`, ...)
- LazyAttribute für `email` (von `username` abgeleitet)
- **Multi-Tenancy Safe**: Generiert schema-spezifische IDs
  - Base Offset: `hash(schema) % 1000 * 1_000_000`
  - Verhindert PK-Kollisionen zwischen Tenant-Schemas
  - Wichtig für Tests mit `django-tenants`

**Schema-Aware ID Generation**:
```python
# Schema "public" → IDs: 1_000_001, 1_000_002, ...
# Schema "tenant1" → IDs: 2_000_001, 2_000_002, ...
# Schema "tenant2" → IDs: 3_000_001, 3_000_002, ...
```

#### InvitationFactory

```python
invitation = InvitationFactory()  # role=GUEST, expires_at=+7d
invitation = InvitationFactory(role=UserProfile.Roles.ADMIN)
invitation = InvitationFactory(expires_at=timezone.now() - timedelta(days=1))  # Expired
```

**Defaults**:
- `email`: Faker-generiert
- `role`: `GUEST`
- `token`: Automatisch via `secrets.token_urlsafe(16)`
- `expires_at`: `now() + 7 Tage`

### Test Cases

#### test_models.py

**Location**: [users/tests/test_models.py](users/tests/test_models.py)

- `test_user_creation`: Basis-Test für UserFactory

#### test_invitation_flow.py

**Location**: [users/tests/test_invitation_flow.py](users/tests/test_invitation_flow.py)

**Tests**:

1. **test_accept_invitation_updates_profile_and_marks_invitation**
   - User akzeptiert Invitation
   - Rolle wird auf Profile übertragen
   - Profile wird aktiviert
   - `accepted_at` wird gesetzt

2. **test_accept_invitation_expired_token_returns_404**
   - Abgelaufener Token führt zu 404
   - Keine Änderung an Profile

3. **test_accept_invitation_cannot_be_used_twice**
   - Erster Zugriff: Erfolgreich (302 Redirect)
   - Zweiter Zugriff: 404 (Token verbraucht)

**Test-Pattern**:
```python
@pytest.mark.django_db
def test_scenario(client):
    # Arrange
    user = UserFactory()
    invitation = InvitationFactory(role=UserProfile.Roles.MANAGER)
    client.force_login(user)

    # Act
    response = client.get(f"/invite/accept/{invitation.token}/")

    # Assert
    assert response.status_code == 302
    invitation.refresh_from_db()
    assert invitation.accepted_at is not None
```

## Workflows

### Neuen Benutzer einladen

**Via Django Admin**:

1. Navigiere zu `/admin/users/invitation/`
2. Klicke "Add Invitation"
3. Fülle aus:
   - E-Mail: `newuser@example.com`
   - Rolle: `GUEST` (oder andere)
4. Speichern (Token bleibt zunächst leer)
5. Invitation in Liste auswählen
6. Action "Gast einladen" ausführen
7. Token wird generiert und in Message angezeigt
8. Link manuell versenden (aktuell)

**Programmatisch**:

```python
from users.models import Invitation
from profiles.models import UserProfile

# Invitation erstellen
invitation = Invitation.objects.create(
    email="newuser@example.com",
    role=UserProfile.Roles.MANAGER
)
token = invitation.generate_token()
invitation.save()

# Link: f"/invite/accept/{token}/"
# E-Mail versenden (TODO: Implementierung)
```

### Einladung akzeptieren

**User-Perspektive**:

1. User erhält E-Mail mit Link `/invite/accept/{token}/`
2. Klick auf Link
3. Falls nicht eingeloggt: Redirect zu Login
4. Nach Login: Automatische Weiterleitung zur Invitation
5. Rolle wird zugewiesen
6. Redirect zu `/` (Dashboard)

**Technischer Flow**:

```
GET /invite/accept/{token}/
  ↓
@login_required Check
  ↓ (nicht eingeloggt)
  ↓ → Redirect /accounts/login/?next=/invite/accept/{token}/
  ↓
  ↓ (eingeloggt)
  ↓
Token validieren (404 wenn ungültig/expired/accepted)
  ↓
UserProfile laden/erstellen
  ↓
profile.role = invitation.role
profile.is_active = True
profile.save()
  ↓
invitation.accepted_at = now()
invitation.save()
  ↓
Redirect /
```

## Sicherheitsaspekte

### Token-Sicherheit

- **Entropie**: 128 Bit (16 Bytes) → 2^128 mögliche Werte
- **URL-Safe**: `secrets.token_urlsafe()` nutzt Base64 ohne `/`, `+`
- **Unique Constraint**: DB verhindert Duplikate
- **One-Time Use**: `accepted_at` macht Token ungültig
- **Expiry**: 7-Tage-Limit reduziert Zeitfenster

**Brute-Force-Schutz**:
- Bei 1 Billion Versuchen/Sekunde: ~10^19 Jahre für 50% Chance
- Praktisch unmöglich ohne Rate-Limiting

### Privilege Escalation

**Verhindert durch**:
1. Login-Pflicht: User muss authentifiziert sein
2. Keine Selbst-Einladung: Admin erstellt Invitations
3. Rolle fix: User kann Rolle nicht wählen
4. One-Time: Keine Mehrfachnutzung für höhere Rollen

**Risiko**: Admin kann beliebige Rollen zuweisen (by design).

### Information Disclosure

- Ungültige/Expired/Accepted Tokens → identische 404
- Kein Unterschied zwischen "existiert nicht" und "bereits verwendet"
- E-Mail-Feld nicht öffentlich sichtbar

## Multi-Tenancy

### Schema-Isolation

User-Tabelle liegt im jeweiligen Tenant-Schema:

```
public.users_user         → Shared/Public Users
tenant1.users_user        → Tenant 1 Users
tenant2.users_user        → Tenant 2 Users
```

**Isolation**:
- Jeder Tenant hat eigene User-Tabelle
- Keine Cross-Tenant User-Zugriffe möglich
- `django-tenants` übernimmt Routing automatisch

### Invitation-Scope

**Aktuelles Verhalten**:
- Invitations sind tenant-spezifisch (liegen im Schema)
- Token ist nur im eigenen Tenant gültig
- Cross-Tenant Invitations nicht möglich

**Zukünftige Erweiterung** (wenn nötig):
- Shared Invitations in `public` Schema
- `tenant_id` Foreign Key für Zuweisung
- Cross-Tenant User-Mapping

## Erweiterungsmöglichkeiten

### E-Mail-Versand

**Aktueller Stand**: Stub in Admin-Action (Print)

**Integration** (z.B. django-anymail):

```python
# settings.py
INSTALLED_APPS += ['anymail']
EMAIL_BACKEND = 'anymail.backends.sendgrid.EmailBackend'
ANYMAIL = {
    'SENDGRID_API_KEY': env('SENDGRID_API_KEY'),
}

# admin.py
from django.core.mail import send_mail

def invite_guest(self, request, queryset):
    for invitation in queryset:
        # ... (Token-Generierung)
        send_mail(
            subject='Einladung zu NOESIS 2',
            message=f'Bitte akzeptieren: {settings.BASE_URL}/invite/accept/{invitation.token}/',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[invitation.email],
        )
```

### Rollen-Permissions

**Aktuell**: Rollen in `UserProfile`, keine Permission-Logik

**Integration** (z.B. django-guardian):

```python
# profiles/models.py
class UserProfile(models.Model):
    # ...

    def has_permission(self, perm):
        return {
            'ADMIN': ['*'],
            'LEGAL': ['view_case', 'edit_case', 'view_document'],
            'MANAGER': ['view_case', 'view_document'],
            'GUEST': ['view_case'],
        }.get(self.role, []).includes(perm)
```

### Invitation-Tracking

**Erweiterung**: Audit-Trail für Invitations

```python
class Invitation(models.Model):
    # ... (bestehende Felder)
    sent_at = models.DateTimeField(null=True, blank=True)
    sent_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    accepted_by = models.ForeignKey(User, null=True, related_name='accepted_invitations', on_delete=models.SET_NULL)
```

### Wiedereinladung

**Feature**: Expired Invitations erneut versenden

```python
def resend_invitation(invitation):
    if invitation.accepted_at:
        raise ValueError("Cannot resend accepted invitation")

    invitation.generate_token()
    invitation.save()
    # E-Mail versenden
```

## Troubleshooting

### Invitation funktioniert nicht

**Symptom**: 404 beim Akzeptieren

**Checks**:
1. Token korrekt? (Case-sensitive, keine Leerzeichen)
2. Bereits akzeptiert? (`accepted_at is not None`)
3. Abgelaufen? (`expires_at < now()`)
4. User eingeloggt? (Login-Pflicht!)

**Debug**:
```python
from users.models import Invitation

inv = Invitation.objects.get(token="...")
print(f"Accepted: {inv.accepted_at}")
print(f"Expires: {inv.expires_at}")
print(f"Is expired: {inv.expires_at < timezone.now() if inv.expires_at else False}")
```

### Profile nicht erstellt

**Symptom**: `UserProfile.DoesNotExist` Exception

**Ursache**: Signal nicht ausgeführt

**Fix**:
```python
from profiles.services import ensure_user_profile

user = User.objects.get(username="...")
profile = ensure_user_profile(user)  # Erstellt falls nicht vorhanden
```

### Multi-Tenancy ID-Kollisionen

**Symptom**: Tests schlagen fehl mit "duplicate key value"

**Ursache**: Factories nutzen nicht schema-spezifische IDs

**Fix**: Verwende `UserFactory` (nicht `User.objects.create`)

```python
# FALSCH (in Multi-Tenant Tests):
user = User.objects.create(username="test")

# RICHTIG:
user = UserFactory(username="test")  # Schema-aware IDs
```

## API (zukünftig)

### REST Endpoints (geplant)

```
POST /api/v1/invitations/        # Admin erstellt Invitation
GET  /api/v1/invitations/{id}/   # Invitation-Details
POST /api/v1/invitations/{id}/send/  # Versenden
POST /api/v1/invitations/{id}/resend/  # Erneut versenden
POST /api/v1/invitations/accept/  # Token akzeptieren (Body: {token})
```

### Contracts (Beispiel)

```python
# POST /api/v1/invitations/
{
  "email": "newuser@example.com",
  "role": "GUEST"
}

# Response 201 Created
{
  "id": 1,
  "email": "newuser@example.com",
  "role": "GUEST",
  "token": null,  # Wird erst bei send/ generiert
  "expires_at": null,
  "accepted_at": null,
  "created_at": "2025-09-10T12:00:00Z"
}
```

## Migrations

### 0001_initial (2025-09-08)

Erstellt `users.User` Modell:
- Erbt alle Felder von `AbstractUser`
- Nutzt `UserManager` von Django
- Setzt `AUTH_USER_MODEL = "users.User"`

**Breaking Change**: Muss VOR allen anderen Apps migriert werden, die auf `AUTH_USER_MODEL` referenzieren.

```bash
python manage.py migrate users 0001
```

### 0002_invitation (2025-09-10)

Fügt `Invitation` Modell hinzu:
- E-Mail + Rolle + Token + Timestamps
- Unique Constraint auf `token`
- Choices für `role` aus `UserProfile.Roles`

**Dependencies**: `users.0001_initial`

```bash
python manage.py migrate users 0002
```

## Konfiguration

### Settings

**Location**: [noesis2/settings/base.py](noesis2/settings/base.py)

```python
# Line 341
AUTH_USER_MODEL = "users.User"

# Login/Logout URLs (Django defaults)
LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"
```

### INSTALLED_APPS

```python
TENANT_APPS = [
    # ...
    'users',
    'profiles',
    'organizations',
]
```

**Wichtig**: `users` muss VOR `profiles` stehen (Dependency).

## Kommandos

### Superuser erstellen

```bash
# Via Make (Multi-Tenancy)
make tenant-superuser SCHEMA=demo USERNAME=admin PASSWORD=secret

# Direkt (Schema-aware)
python manage.py tenant_command createsuperuser --schema=demo
```

### Invitations verwalten

**Django Shell** (tenant-aware):

```python
python manage.py tenant_command shell --schema=demo

from users.models import Invitation
from profiles.models import UserProfile

# Alle offenen Invitations
Invitation.objects.filter(accepted_at__isnull=True)

# Abgelaufene Invitations
from django.utils import timezone
Invitation.objects.filter(
    expires_at__lt=timezone.now(),
    accepted_at__isnull=True
)

# Neue Invitation
inv = Invitation.objects.create(
    email="new@example.com",
    role=UserProfile.Roles.GUEST
)
inv.generate_token()
inv.save()
print(f"Token: {inv.token}")
```

## Dependencies

**Python Packages**:
- `django>=5.2.6`: Framework
- `django-tenants`: Multi-Tenancy (Schema-Isolation)
- `factory-boy`: Test Factories
- `pytest-django`: Testing Framework

**Internal Apps**:
- `profiles`: UserProfile mit Rollen & `ensure_user_profile()`
- `organizations`: Organisation-Membership (optional)

**External Services**:
- Zukünftig: E-Mail-Provider (SendGrid, Mailgun, etc.)

## Weiterführende Dokumentation

- **Multi-Tenancy**: [docs/multi-tenancy.md](../docs/multi-tenancy.md)
- **Profiles App**: [profiles/README.md](../profiles/README.md) (TODO)
- **Django Auth**: https://docs.djangoproject.com/en/5.2/topics/auth/
- **django-tenants**: https://django-tenants.readthedocs.io/

## Changelog

### 2025-09-10
- ✅ Invitation Model mit Token-System
- ✅ Admin-Action für Einladungsversand (Stub)
- ✅ Integration mit UserProfile-Rollen
- ✅ Tests für kompletten Invitation-Flow

### 2025-09-08
- ✅ Initial Migration mit Custom User Model
- ✅ AUTH_USER_MODEL Konfiguration
- ✅ UserFactory mit Multi-Tenancy Support

## TODOs

- [ ] E-Mail-Versand implementieren (django-anymail)
- [ ] REST API für Invitation-Management
- [ ] Bulk-Import von Invitations (CSV)
- [ ] Invitation-Templates (E-Mail-HTML)
- [ ] Wiedereinladungs-Feature (resend)
- [ ] Admin-Notification bei Invitation-Akzeptierung
- [ ] Rate-Limiting für Token-Checks
- [ ] Audit-Log für Invitation-Events

---

**Version**: 1.0
**Zuletzt aktualisiert**: 2025-12-15
**Maintainer**: NOESIS 2 Team
