from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0002_documentingestionrun_collection_id_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="documentingestionrun",
            old_name="tenant",
            new_name="tenant_id",
        ),
        migrations.RemoveIndex(
            model_name="documentingestionrun",
            name="documents_d_tenant_2901aa_idx",
        ),
        migrations.RemoveIndex(
            model_name="documentingestionrun",
            name="documents_d_tenant_2c4d54_idx",
        ),
        migrations.RemoveIndex(
            model_name="documentlifecyclestate",
            name="documents_d_tenant__816925_idx",
        ),
        migrations.RemoveConstraint(
            model_name="documentingestionrun",
            name="document_ingestion_run_unique_case",
        ),
        migrations.AddConstraint(
            model_name="documentingestionrun",
            constraint=models.UniqueConstraint(
                fields=("tenant_id", "case"),
                name="document_ingestion_run_unique_case",
            ),
        ),
        migrations.AddIndex(
            model_name="documentlifecyclestate",
            index=models.Index(
                fields=("tenant_id", "workflow_id"),
                name="doc_lifecycle_tenant_wf_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="documentingestionrun",
            index=models.Index(
                fields=("tenant_id", "case"),
                name="doc_ing_run_tenant_case_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="documentingestionrun",
            index=models.Index(
                fields=("tenant_id", "run_id"),
                name="doc_ing_run_tenant_run_idx",
            ),
        ),
    ]
