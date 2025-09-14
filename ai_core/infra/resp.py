from __future__ import annotations

from django.http import HttpResponse


def apply_std_headers(
    response: HttpResponse, trace_id: str, prompt_version: str | None = None
) -> HttpResponse:
    """Attach standard headers to a response.

    Parameters
    ----------
    response:
        The response object to modify.
    trace_id:
        Trace identifier for linking logs and metrics.
    prompt_version:
        Version identifier for the prompt used to generate the response. Only
        set when provided.
    """

    response["X-Trace-ID"] = trace_id
    if prompt_version:
        response["X-Prompt-Version"] = prompt_version
    return response
