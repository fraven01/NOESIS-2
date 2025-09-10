from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("projects", "0002_project_organization"),
    ]

    operations = [
        migrations.DeleteModel(
            name="WorkflowInstance",
        ),
    ]
