from django.contrib.auth import login, get_user_model
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny

from profiles.services import ensure_user_profile
from .models import Invitation

User = get_user_model()


@permission_classes([AllowAny])
def accept_invitation(request, token):
    """Accept invitation - creates user if needed, or updates existing user's profile."""
    invitation = get_object_or_404(Invitation, token=token)

    # Check expiry
    if invitation.accepted_at:
        raise PermissionDenied("Invitation already accepted")
    if (
        invitation.invitation_expires_at
        and invitation.invitation_expires_at < timezone.now()
    ):
        raise PermissionDenied("Invitation expired")

    # Case 1: User is authenticated - just update their profile
    if request.user.is_authenticated:
        invited_email = invitation.email.strip().lower()
        user_email = (request.user.email or "").strip().lower()
        if user_email != invited_email:
            raise PermissionDenied("Invitation not for this user")
        profile = ensure_user_profile(request.user)
        profile.role = invitation.role
        profile.account_type = invitation.account_type
        profile.expires_at = invitation.user_expires_at
        profile.is_active = True
        profile.save()
        invitation.accepted_at = timezone.now()
        invitation.save()
        return redirect("/")

    # Case 2: User needs to be created (EXTERNAL accounts or new users)
    if request.method == "POST":
        # Extract password, create user, log them in
        password = request.POST.get("password")
        password_confirm = request.POST.get("password_confirm")

        existing_user = User.objects.filter(email__iexact=invitation.email).first()
        if existing_user:
            return render(
                request,
                "registration/set_password.html",
                {
                    "invitation": invitation,
                    "error": "An account already exists for this email. Please log in to accept the invitation.",
                },
            )

        if not password or password != password_confirm:
            return render(
                request,
                "registration/set_password.html",
                {
                    "invitation": invitation,
                    "error": "Passwords don't match or are empty",
                },
            )

        # Create user (normalize email for consistency)
        email_normalized = invitation.email.strip().lower()
        user = User.objects.create_user(
            username=email_normalized,
            email=email_normalized,
            password=password,
        )

        # Create profile
        profile = ensure_user_profile(user)
        profile.role = invitation.role
        profile.account_type = invitation.account_type
        profile.expires_at = invitation.user_expires_at
        profile.is_active = True
        profile.save()

        # Mark invitation accepted
        invitation.accepted_at = timezone.now()
        invitation.save()

        # Log user in
        login(request, user)
        return redirect("/")

    # GET: Show password set form for new users
    return render(
        request,
        "registration/set_password.html",
        {
            "invitation": invitation,
        },
    )
