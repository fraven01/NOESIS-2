from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.deprecation import MiddlewareMixin
from structlog.stdlib import get_logger

logger = get_logger(__name__)


class SimulatedUserMiddleware(MiddlewareMixin):
    """
    Allow staff users to simulate other users in the RAG Workbench (DEBUG only).

    Security:
    - This middleware only runs if DEBUG=True or RAG_TOOLS_ENABLED=True.
    - It strictly requires the ACTING user to be authenticated and have `is_staff=True`.
    - It preserves the actual user as `request.original_user`.
    """

    def process_request(self, request):
        if not (settings.DEBUG or getattr(settings, "RAG_TOOLS_ENABLED", False)):
            return

        if not request.path.startswith("/rag-tools/"):
            return

        # SECURITY: Ensure the acting user is known and privileged.
        # We rely on AuthenticationMiddleware having run before this.
        if not hasattr(request, "user") or not request.user.is_authenticated:
            return

        if not request.user.is_staff:
            return

        simulated_user_id = request.session.get("rag_tools_simulated_user_id")
        if not simulated_user_id:
            return

        # Allow passing "anonymous" as a string to simulate an unauthenticated user
        if simulated_user_id == "anonymous":
            request.original_user = request.user
            from django.contrib.auth.models import AnonymousUser

            request.user = AnonymousUser()
            request.is_simulated_user = True
            return

        User = get_user_model()
        try:
            target_user = User.objects.get(pk=simulated_user_id)
            # Store the real user before switching
            request.original_user = request.user
            request.user = target_user
            request.is_simulated_user = True
        except User.DoesNotExist:
            logger.warning(
                "rag_tools.simulation.user_not_found", user_id=simulated_user_id
            )
            # Clean up invalid session key
            if "rag_tools_simulated_user_id" in request.session:
                del request.session["rag_tools_simulated_user_id"]
        except Exception:
            logger.exception("rag_tools.simulation.failed")
