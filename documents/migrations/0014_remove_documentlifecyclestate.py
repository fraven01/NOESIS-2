# Generated manually to remove DocumentLifecycleState model

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0013_alter_documentlifecyclestate_tenant_id_and_more"),
    ]

    operations = [
        # Drop the DocumentLifecycleState table
        migrations.RunSQL(
            sql="DROP TABLE IF EXISTS documents_documentlifecyclestate CASCADE;",
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
