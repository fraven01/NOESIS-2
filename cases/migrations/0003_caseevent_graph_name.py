from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cases", "0002_backfill_case_records"),
    ]

    operations = [
        migrations.AddField(
            model_name="caseevent",
            name="graph_name",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
    ]
