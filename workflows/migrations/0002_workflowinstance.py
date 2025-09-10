import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflows", "0001_initial"),
        ("projects", "0002_project_organization"),
    ]

    operations = [
        migrations.CreateModel(
            name="WorkflowInstance",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("draft", "Draft"),
                            ("review", "Review"),
                            ("final", "Final"),
                        ],
                        default="draft",
                        max_length=10,
                    ),
                ),
                (
                    "project",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="workflows",
                        to="projects.project",
                    ),
                ),
            ],
            options={"abstract": False},
        ),
    ]
