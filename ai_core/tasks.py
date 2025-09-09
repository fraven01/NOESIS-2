from celery import shared_task


@shared_task
def process_document_task(document_id):
    print(f"Starte Verarbeitung für Dokument {document_id}")
