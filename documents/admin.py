from django.contrib import admin

from .models import Document, DocumentType


admin.site.register(DocumentType)
admin.site.register(Document)
