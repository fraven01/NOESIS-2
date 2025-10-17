from django.db import migrations


class Migration(migrations.Migration):
    dependencies = []

    operations = [
        migrations.RunSQL(
            sql="""
                DROP TABLE IF EXISTS workflows_workflowinstance CASCADE;
                DROP TABLE IF EXISTS workflows_workflowstep CASCADE;
                DROP TABLE IF EXISTS workflows_workflowtemplate CASCADE;
            """,
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
