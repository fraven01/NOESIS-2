from rest_framework.routers import SimpleRouter

from .dev_api import CollectionDevViewSet, DocumentDevViewSet

router = SimpleRouter()
router.register(r"documents", DocumentDevViewSet, basename="dev-documents")
router.register(r"collections", CollectionDevViewSet, basename="dev-collections")

urlpatterns = router.urls
