from django.db import migrations, models
from django.utils import timezone


def forwards(apps, schema_editor):
    Tenant = apps.get_model("customers", "Tenant")
    today = timezone.now().date()
    # Backfill created_on for existing rows where it's NULL
    Tenant.objects.filter(created_on__isnull=True).update(created_on=today)


def backwards(apps, schema_editor):
    # No special backward data migration needed; fields are removed by schema ops.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("customers", "0003_domain_unique_primary_domain_per_tenant"),
    ]

    operations = [
        migrations.AddField(
            model_name="tenant",
            name="paid_until",
            field=models.DateField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="tenant",
            name="on_trial",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="tenant",
            name="created_on",
            field=models.DateField(auto_now_add=True, null=True),
        ),
        migrations.RunPython(forwards, backwards),
        migrations.AlterField(
            model_name="tenant",
            name="created_on",
            field=models.DateField(auto_now_add=True),
        ),
    ]

