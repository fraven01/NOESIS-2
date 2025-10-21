"""Forms used by the RAG operations workbench."""

from __future__ import annotations

import json
from typing import Iterable, Sequence

from django import forms
from django.conf import settings

from ai_core.rag.embedding_config import get_embedding_configuration


def _clean_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _available_embedding_profiles() -> Sequence[tuple[str, str]]:
    try:
        config = get_embedding_configuration()
    except Exception:
        config = None

    choices: list[tuple[str, str]]
    if config is not None:
        choices = [
            (profile_id, profile_id)
            for profile_id in sorted(config.embedding_profiles.keys())
        ]
    else:
        choices = []

    default_profile = getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", None)
    if default_profile:
        default_pair = (str(default_profile), str(default_profile))
        if default_pair not in choices:
            choices = [default_pair, *[choice for choice in choices if choice != default_pair]]

    if not choices:
        choices = [("standard", "standard")]

    return choices


class RagUploadForm(forms.Form):
    """Handle manual document uploads triggered from the workbench."""

    file = forms.FileField(label="Datei")
    case_id = forms.CharField(
        label="Case ID",
        initial="manual-workbench",
        required=True,
        help_text="Wird für Upload und Ingestion als X-Case-ID verwendet.",
    )
    collection_id = forms.CharField(
        label="Collection",
        required=False,
        help_text="Optional: Überschreibt die Resolver-Collection.",
    )
    metadata = forms.CharField(
        label="Metadata (optional)",
        required=False,
        widget=forms.Textarea(attrs={"rows": 4}),
        help_text="JSON-Objekt, wird mit Collection und External-ID angereichert.",
    )

    def clean_case_id(self) -> str:
        value = _clean_identifier(self.cleaned_data.get("case_id"))
        if not value:
            raise forms.ValidationError("Case ID darf nicht leer sein.")
        return value

    def clean_collection_id(self) -> str | None:
        return _clean_identifier(self.cleaned_data.get("collection_id"))

    def clean_metadata(self) -> dict[str, object]:
        raw = self.cleaned_data.get("metadata")
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise forms.ValidationError(
                "Metadata muss gültiges JSON sein."
            ) from exc
        if not isinstance(payload, dict):
            raise forms.ValidationError("Metadata muss ein JSON-Objekt sein.")
        return payload


class RagIngestionForm(forms.Form):
    """Queue ingestion runs for existing document IDs."""

    case_id = forms.CharField(
        label="Case ID",
        initial="manual-workbench",
        required=True,
        help_text="X-Case-ID für den Ingestion-Run.",
    )
    document_ids = forms.CharField(
        label="Document IDs",
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Kommagetrennt oder zeilenweise.",
    )
    embedding_profile = forms.ChoiceField(
        label="Embedding Profile",
        choices=_available_embedding_profiles(),
    )
    collection_id = forms.CharField(
        label="Collection",
        required=False,
        help_text="Optional: Persistiert Collection-Scope für die Dokumente.",
    )

    def clean_case_id(self) -> str:
        value = _clean_identifier(self.cleaned_data.get("case_id"))
        if not value:
            raise forms.ValidationError("Case ID darf nicht leer sein.")
        return value

    def clean_document_ids(self) -> list[str]:
        raw = self.cleaned_data.get("document_ids", "")
        candidates: Iterable[str] = []
        if isinstance(raw, str):
            separators = [",", "\n", "\r"]
            for separator in separators:
                raw = raw.replace(separator, " ")
            candidates = raw.split()
        normalised = [value.strip() for value in candidates if value.strip()]
        if not normalised:
            raise forms.ValidationError("Mindestens eine Document ID ist erforderlich.")
        return normalised

    def clean_collection_id(self) -> str | None:
        return _clean_identifier(self.cleaned_data.get("collection_id"))


class RagStatusForm(forms.Form):
    """Read the latest ingestion status for a case."""

    case_id = forms.CharField(
        label="Case ID",
        initial="manual-workbench",
        required=True,
        help_text="X-Case-ID des letzten Ingestion-Runs.",
    )

    def clean_case_id(self) -> str:
        value = _clean_identifier(self.cleaned_data.get("case_id"))
        if not value:
            raise forms.ValidationError("Case ID darf nicht leer sein.")
        return value


class RagQueryForm(forms.Form):
    """Execute retrieval augmented queries through the backend graph."""

    case_id = forms.CharField(
        label="Case ID",
        initial="manual-workbench",
        required=True,
        help_text="X-Case-ID für den Query-Lauf.",
    )
    query = forms.CharField(
        label="Query",
        widget=forms.Textarea(attrs={"rows": 3}),
    )
    process = forms.CharField(
        label="Process",
        required=False,
    )
    collection_id = forms.CharField(
        label="Collection",
        required=False,
    )
    visibility = forms.ChoiceField(
        label="Visibility",
        required=False,
        choices=(
            ("", "Default"),
            ("active", "active"),
            ("all", "all"),
            ("deleted", "deleted"),
        ),
    )
    alpha = forms.FloatField(
        label="alpha",
        required=False,
        min_value=0.0,
        max_value=1.0,
    )
    min_sim = forms.FloatField(
        label="min_sim",
        required=False,
        min_value=0.0,
        max_value=1.0,
    )
    top_k = forms.IntegerField(
        label="top_k",
        required=False,
        min_value=1,
    )
    vec_limit = forms.IntegerField(
        label="vec_limit",
        required=False,
        min_value=1,
    )
    lex_limit = forms.IntegerField(
        label="lex_limit",
        required=False,
        min_value=1,
    )
    max_candidates = forms.IntegerField(
        label="max_candidates",
        required=False,
        min_value=1,
    )
    trgm_limit = forms.FloatField(
        label="trgm_limit",
        required=False,
        min_value=0.0,
        max_value=1.0,
    )

    def clean_case_id(self) -> str:
        value = _clean_identifier(self.cleaned_data.get("case_id"))
        if not value:
            raise forms.ValidationError("Case ID darf nicht leer sein.")
        return value

    def clean_collection_id(self) -> str | None:
        return _clean_identifier(self.cleaned_data.get("collection_id"))

    def build_payload(self) -> dict[str, object]:
        if not self.is_valid():  # pragma: no cover - guard for defensive use
            raise ValueError("RagQueryForm must be valid before building payload")

        payload: dict[str, object] = {
            "question": self.cleaned_data["query"].strip(),
            "query": self.cleaned_data["query"].strip(),
        }

        optional_fields = (
            "process",
            "collection_id",
            "visibility",
            "alpha",
            "min_sim",
            "top_k",
            "vec_limit",
            "lex_limit",
            "max_candidates",
            "trgm_limit",
        )

        for field in optional_fields:
            value = self.cleaned_data.get(field)
            if value in (None, ""):
                continue
            payload[field] = value

        return payload


__all__ = [
    "RagUploadForm",
    "RagIngestionForm",
    "RagStatusForm",
    "RagQueryForm",
]
