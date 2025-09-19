NOESIS 2 – GKE Worker/Beat Deployment

Dieses Verzeichnis enthält Kubernetes‑Manifeste und Hinweise, um Celery Worker und Beat stabil auf GKE (Autopilot) zu betreiben. Web/LiteLLM/Langfuse bleiben auf Cloud Run.

Übersicht
- Namespace: `noesis2`
- ServiceAccount: `noesis2-worker-sa` (Workload Identity → GSA mit Cloud SQL Client)
- Deployments: `celery-worker`, `celery-beat`
- Sidecar: Cloud SQL Auth Proxy 2.x (localhost:5432)
- HPA: CPU‑basiert für Worker (1–5 Replicas)

Voraussetzungen
- GKE Autopilot Cluster in derselben VPC wie Cloud SQL + Memorystore (private IPs erreichbar)
- gcloud und kubectl lokal/CI verfügbar
- Cloud SQL Instanz + Datenbanken (App‑DB, optional LiteLLM/Langfuse DBs)
- Memorystore Redis (Private IP)

Workload Identity – ServiceAccount binden
1) Erzeuge/verwende eine GCP‑Service‑Account (GSA) mit Rolle „Cloud SQL Client“:
   gcloud iam service-accounts create noesis2-workers \
     --project YOUR_PROJECT \
     --display-name "NOESIS 2 Workers"
   gcloud projects add-iam-policy-binding YOUR_PROJECT \
     --member "serviceAccount:noesis2-workers@YOUR_PROJECT.iam.gserviceaccount.com" \
     --role roles/cloudsql.client
2) Trage die GSA in `k8s/serviceaccount.yaml` ein (Annotation `iam.gke.io/gcp-service-account`).
3) GKE‑Credentials abrufen und Namespace/SA anwenden:
   gcloud container clusters get-credentials CLUSTER_NAME --region LOCATION --project YOUR_PROJECT
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/serviceaccount.yaml

Secrets anlegen
- Passe `k8s/secret-example.yaml` auf eure Werte an und lege das Secret an:
  kubectl apply -f k8s/secret-example.yaml
- Datenbankname in den Deployments (Value von `DATABASE_URL`) auf eure DB setzen (z. B. `noesis2`).
 
Deployments anwenden
- Ersetze Platzhalter und apply:
  export IMAGE="REGION-docker.pkg.dev/YOUR_PROJECT/REPO/noesis2-web:SHA"
  export INSTANCE="YOUR_PROJECT:REGION:INSTANCE"  # CLOUD_SQL_CONNECTION_NAME
  sed -e "s#REGION-docker.pkg.dev/PROJECT/REPO/noesis2-web:SHA#${IMAGE}#g" \
      -e "s#PROJECT:REGION:INSTANCE#${INSTANCE}#g" \
      k8s/worker-deployment.yaml | kubectl apply -f -
  sed -e "s#REGION-docker.pkg.dev/PROJECT/REPO/noesis2-web:SHA#${IMAGE}#g" \
      -e "s#PROJECT:REGION:INSTANCE#${INSTANCE}#g" \
      k8s/beat-deployment.yaml | kubectl apply -f -
  kubectl apply -f k8s/worker-hpa.yaml
- Rollout prüfen:
  kubectl rollout status deploy/celery-worker -n noesis2 --timeout=120s
  kubectl rollout status deploy/celery-beat -n noesis2 --timeout=120s

Smoke‑Checks
- DB/Redis Konnektivität (einfacher Pod‑Exec):
  kubectl exec -n noesis2 deploy/celery-worker -- sh -lc "pgrep -f 'celery.*worker' >/dev/null && echo ok"
  # Python DB‑Ping (wenn psycopg2 im Image vorhanden ist)
  kubectl exec -n noesis2 deploy/celery-worker -- sh -lc "python - <<'PY'\nimport os, psycopg2; d=os.environ['DATABASE_URL']; print('db ok')\nPY"

Tuning & Betrieb
- Probes: pgrep‑basiert (leichtgewichtig). Optional `celery inspect ping` (teurer).
- Graceful Shutdown: `terminationGracePeriodSeconds` + preStop‑Hooks sind gesetzt.
- HPA: CPU‑basiert (1–5). Perspektivisch Queue‑Länge als Custom Metric nutzen.
- Cloud SQL Proxy: Sidecar mit `--private-ip`. Alternativ direkte Private‑IP Verbindung (Proxy entfällt), dann `DATABASE_URL` auf Instanz‑IP anpassen.

CI/CD
- GitHub Actions enthält einen Job `deploy-gke-workers`, der die Manifeste anwendet und Platzhalter ersetzt. Voraussetzungen als Repo‑Secrets:
  - `GKE_CLUSTER_NAME`, `GKE_LOCATION`, `GCP_PROJECT_ID`, `GCP_REGION`, `GAR_REPOSITORY`, `CLOUD_SQL_CONNECTION_NAME`.

Hinweise Multitenancy
- Tenant‑Kontext wird in Tasks übergeben (z. B. Schema/ID in Task args). Datenbankverbindung entspricht dem Web‑Service (django‑tenants). Redis/DB zeigen auf die gemeinsamen Services in derselben VPC.

