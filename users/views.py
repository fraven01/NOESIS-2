from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect
from django.utils import timezone

from profiles.services import ensure_user_profile
from .models import Invitation


@login_required
def accept_invitation(request, token):
    invitation = get_object_or_404(Invitation, token=token)
    if invitation.accepted_at or (
        invitation.expires_at and invitation.expires_at < timezone.now()
    ):
        raise Http404
    profile = ensure_user_profile(request.user)
    profile.role = invitation.role
    profile.is_active = True
    profile.save()
    invitation.accepted_at = timezone.now()
    invitation.save()
    return redirect("/")
