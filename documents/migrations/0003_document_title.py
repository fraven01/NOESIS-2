from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0002_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="title",
            field=models.CharField(max_length=255, default=""),
            preserve_default=False,
        ),
    ]
