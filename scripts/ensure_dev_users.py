import os
import django


def ensure_dev_users():
    from django.contrib.auth import get_user_model
    from profiles.models import UserProfile

    User = get_user_model()

    users_data = [
        {
            "username": "admin",
            "email": "admin@example.com",
            "first_name": "Admin",
            "last_name": "User",
            "role": UserProfile.Roles.TENANT_ADMIN,
        },
        {
            "username": "legal_bob",
            "email": "bob@example.com",
            "first_name": "Bob",
            "last_name": "Legal",
            "role": UserProfile.Roles.LEGAL,
        },
        {
            "username": "alice_stakeholder",
            "email": "alice@example.com",
            "first_name": "Alice",
            "last_name": "Stakeholder",
            "role": UserProfile.Roles.STAKEHOLDER,
        },
        {
            "username": "charles_external",
            "email": "charles@external.com",
            "first_name": "Charles",
            "last_name": "External",
            "role": UserProfile.Roles.STAKEHOLDER,
            "account_type": UserProfile.AccountType.EXTERNAL,
        },
    ]

    print("Ensuring dev users exist...")

    for u_data in users_data:
        username = u_data["username"]
        role = u_data.get("role")
        account_type = u_data.get("account_type", UserProfile.AccountType.INTERNAL)

        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "email": u_data["email"],
                "first_name": u_data["first_name"],
                "last_name": u_data["last_name"],
                "is_active": True,
                "is_staff": username == "admin",
                "is_superuser": username == "admin",
            },
        )

        if created:
            user.set_password("password")
            user.save()
            print(f"Created user: {username}")
        else:
            print(f"User exists: {username}")

        # Ensure profile
        profile, p_created = UserProfile.objects.get_or_create(user=user)
        if p_created or profile.role != role or profile.account_type != account_type:
            profile.role = role
            profile.account_type = account_type
            profile.save()
            print(f"Updated profile for: {username} (Role: {role})")


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")
    django.setup()
    ensure_dev_users()
