set -euo pipefail

# Hinweis: Cloud Run (LiteLLM) läuft in der EU-Region, Vertex AI bleibt in us-central1.
# Zugriffe von LiteLLM auf Vertex AI verlassen damit die EU-Region (technisch erforderlich).

# === Variablen anpassen ===
PROJECT_ID="noesis-2-staging"  # z.B. mein-projekt-123456
REGION="europe-west3"            # Cloud Run Region (DSGVO)
SQL_INSTANCE="noesis-2-staging:europe-west3:noesis-2-staging-db"
SA_NAME="litellm-proxy"                 # Serviceaccount-Name
VERTEX_LOCATION="us-central1"           # Vertex AI Region (z. B. für Gemini)

# === Preflight: gcloud & optional .env.gcloud übernehmen ===
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI nicht gefunden. Bitte installieren und authentisieren (gcloud auth login)." >&2
  exit 1
fi

# Optional: .env.gcloud einlesen, falls vorhanden
if [ -f ./.env.gcloud ]; then
  # Nur die gewünschten Keys lesen (keine Spaces erwartet)
  while IFS='=' read -r k v; do
    case "$k" in
      GCP_PROJECT) PROJECT_ID="${PROJECT_ID:-$v}" ;;
      GCP_REGION) REGION="${REGION:-$v}" ;;
      GCP_SQL_CONNECTION_NAME) SQL_INSTANCE="${SQL_INSTANCE:-$v}" ;;
    esac
  done < <(grep -E '^(GCP_PROJECT|GCP_REGION|GCP_SQL_CONNECTION_NAME)=' ./.env.gcloud | sed 's/\r$//')
fi

# Aktives Konto prüfen
ACTIVE_ACCT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' || true)"
if [ -z "${ACTIVE_ACCT:-}" ]; then
  echo "Kein aktives gcloud-Konto. Bitte 'gcloud auth login' oder 'gcloud auth activate-service-account' ausführen." >&2
  exit 1
fi

# Platzhalter prüfen
if echo "$SQL_INSTANCE" | grep -q '<PROJECT_ID\|<REGION\|<INSTANCE_NAME>'; then
  echo "Bitte SQL_INSTANCE setzen (z. B. $PROJECT_ID:$REGION:<INSTANCE_NAME> oder Connection-Name aus .env.gcloud)." >&2
  exit 1
fi

# === Projekt setzen ===
gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"
gcloud config set compute/region "$REGION"
gcloud config set ai/region "$VERTEX_LOCATION"

# Sichtprüfung der aktiven Defaults
echo "gcloud project: $(gcloud config get-value project)"
echo "gcloud run/region: $(gcloud config get-value run/region)"
echo "gcloud compute/region: $(gcloud config get-value compute/region)"
echo "gcloud ai/region: $(gcloud config get-value ai/region)"

# === APIs aktivieren (idempotent) ===
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  sqladmin.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  secretmanager.googleapis.com

# === Service Account anlegen oder wiederverwenden ===
if ! gcloud iam service-accounts describe "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="LiteLLM Proxy Runtime"
fi
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Ensure SA is fully propagated before binding roles (eventual consistency)
echo "Warte auf Service Account Propagation: $SA_EMAIL"
for i in $(seq 1 20); do
  if gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
    echo "Service Account gefunden (Versuch $i)"; break
  fi
  sleep 3
done
if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  echo "Service Account $SA_EMAIL nicht auffindbar. Bitte erneut ausführen oder IAM Konsistenz prüfen." >&2
  exit 1
fi

# === Rollen für den RUNTIME-SA (Cloud Run) ===
# Vertex AI Zugriff
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"

# Cloud SQL Connector
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/cloudsql.client"

# Artifact Registry Images ziehen
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.reader"

# Zugriff auf Secret Manager für API-Keys (z. B. GEMINI_API_KEY)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"

# (Optional) Logs schreiben explizit erlauben
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter" || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/monitoring.metricWriter" || true

# === Cloud Run Service auf SA umstellen (falls schon deployed) ===
# Kann auch erst nach dem ersten Deploy laufen.
if gcloud run services describe litellm-proxy --region "$REGION" --format='value(metadata.name)' >/dev/null 2>&1; then
  gcloud run services update litellm-proxy \
    --region "$REGION" \
    --service-account "$SA_EMAIL"
fi

# === Vertex-Region ist nur Laufzeit-ENV ===
# In deiner Pipeline bleibt: VERTEXAI_PROJECT=$PROJECT_ID, VERTEXAI_LOCATION=$VERTEX_LOCATION

# === Checks ===
echo "== Enabled APIs =="
gcloud services list --enabled | grep -E 'aiplatform|run|artifact|sqladmin' || true

echo "== SA Rollen =="
gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_EMAIL}" \
  --format='table(bindings.role)'

echo "== Optional: Cloud SQL Reachability =="
echo "Cloud SQL: $SQL_INSTANCE (Connector wird zur Laufzeit via --add-cloudsql-instances genutzt)"

echo "== Summary =="
echo "Project.............: $PROJECT_ID"
echo "Run/Region..........: $REGION"
echo "Compute/Region......: $(gcloud config get-value compute/region)"
echo "Vertex AI Region....: $VERTEX_LOCATION"
echo "Service Account.....: $SA_EMAIL"
echo "SQL Instance........: $SQL_INSTANCE"
echo "APIs enabled........: aiplatform, run, artifactregistry, sqladmin, logging, monitoring, secretmanager"
echo "Script abgeschlossen. Für das erste Deploy übernimmt die CI-Pipeline den Cloud Run Deploy inkl. ENV Vars."
