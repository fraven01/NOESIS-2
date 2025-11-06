"""Error response helpers."""
from django.http import JsonResponse


def error(status: int, code: str, message: str, details: dict = None) -> JsonResponse:
    """Standardized error response."""
    payload = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return JsonResponse(payload, status=status)
