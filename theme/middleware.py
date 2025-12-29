from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.deprecation import MiddlewareMixin
from structlog.stdlib import get_logger

logger = get_logger(__name__)


class SimulatedUserMiddleware(MiddlewareMixin):
    """
    Middleware that allows overriding the logged-in user for 'rag-tools' views
    to simulate different personas (Admin, Legal, etc.) during development.

    This is STRICTLY for development and requires DEBUG=True or RAG_TOOLS_ENABLED=True.
    """

    def process_request(self, request):
        if not (settings.DEBUG or getattr(settings, "RAG_TOOLS_ENABLED", False)):
            return

        if not request.path.startswith("/rag-tools/"):
            return

        simulated_user_id = request.session.get("rag_tools_simulated_user_id")
        if not simulated_user_id:
            return

        User = get_user_model()
        try:
            # We fetch the user instance. We handle the case where it might be a string ID or int.
            user = User.objects.get(pk=simulated_user_id)
            # Override request.user.
            # Note: This does NOT log the user in via session auth,
            # it just overrides the attribute for this request cycle.
            request.user = user
            request.is_simulated_user = True

            # Also attach to scope context if possible/needed, or just rely on
            # views extracting it from request.user

        except User.DoesNotExist:
            logger.warning(
                "rag_tools.simulation.user_not_found", user_id=simulated_user_id
            )
            # Clean up invalid session key
            if "rag_tools_simulated_user_id" in request.session:
                del request.session["rag_tools_simulated_user_id"]
        except Exception:
            logger.exception("rag_tools.simulation.failed")
