from django.contrib import admin

from .models import Workflow, WorkflowStep


admin.site.register(Workflow)
admin.site.register(WorkflowStep)
