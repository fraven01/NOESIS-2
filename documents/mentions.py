"""Helpers for parsing @mentions in comment text."""

from __future__ import annotations

import re
from typing import Iterable
from uuid import UUID

from django.contrib.auth import get_user_model
from django.db.models.functions import Lower


RICH_MENTION_PATTERN = re.compile(r"<@(?P<user_id>[0-9a-fA-F-]{32,36})>")
USERNAME_PATTERN = re.compile(r"(?<![\w@])@(?P<username>[A-Za-z0-9_.-]{3,})")


def _unique_users(users: Iterable[object]) -> list[object]:
    seen = set()
    ordered = []
    for user in users:
        user_id = getattr(user, "id", None)
        if user_id is None or user_id in seen:
            continue
        seen.add(user_id)
        ordered.append(user)
    return ordered


def resolve_mentioned_users(text: str | None) -> list[object]:
    """Resolve mention targets from rich tokens and fallback @username."""

    if not text:
        return []

    User = get_user_model()

    rich_user_ids: set[str] = set()
    for match in RICH_MENTION_PATTERN.finditer(text):
        raw_user_id = match.group("user_id")
        if not raw_user_id:
            continue
        try:
            rich_user_ids.add(str(UUID(raw_user_id)))
        except (TypeError, ValueError):
            continue

    rich_users = list(User.objects.filter(id__in=rich_user_ids))

    scrubbed = RICH_MENTION_PATTERN.sub(" ", text)
    fallback_usernames = {
        match.group("username")
        for match in USERNAME_PATTERN.finditer(scrubbed)
        if match.group("username")
    }

    fallback_users = []
    if fallback_usernames:
        lowered_usernames = {username.lower() for username in fallback_usernames}
        candidates = (
            User.objects.annotate(username_lower=Lower("username"))
            .filter(username_lower__in=lowered_usernames)
            .values("id", "username_lower")
        )
        by_username: dict[str, list[str]] = {}
        for candidate in candidates:
            by_username.setdefault(candidate["username_lower"], []).append(
                str(candidate["id"])
            )

        unique_ids = [ids[0] for ids in by_username.values() if len(ids) == 1]
        if unique_ids:
            fallback_users = list(User.objects.filter(id__in=unique_ids))

    return _unique_users([*rich_users, *fallback_users])
