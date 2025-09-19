#!/usr/bin/env bash
set -euo pipefail

REGION=""
ZONE=""
OUT_FILE=".env.gcloud"
FETCH_SECRETS=0
SECRETS=()
# CI/GitHub Actions service account detection
CI_GSA_EMAIL=""
CI_CRED_FILE=""
DETECT_CI_GSA=1

usage() {
  cat <<'USAGE'
Usage: scripts/gcloud-bootstrap.sh [options]

Options:
  --region REGION          Preferred GCP region (e.g., europe-west3)
  --zone ZONE              Preferred GCP zone (e.g., europe-west3-a)
  -o, --out FILE           Output env file (default: .env.gcloud)
  --fetch-secrets          Fetch secret payloads (disabled by default)
  --secret NAME            Secret Manager name to fetch (repeatable)
  --ci-gsa EMAIL           Explicit CI/GitHub Actions service account email
  --ci-cred-file FILE      Path to SA JSON (extracts client_email)
  --no-ci-detect           Skip CI/GSA auto-detection
  -h, --help               Show this help

This script authenticates gcloud, lets you choose a project and region,
discovers Redis (Memorystore), Cloud SQL (Postgres), Cloud Run URLs and
Artifact Registry repos, then writes a safe .env-like file.
Secrets are NOT fetched unless --fetch-secrets and --secret are provided.
USAGE
}

info() { printf "[i] %s\n" "$*"; }
warn() { printf "[!] %s\n" "$*" 1>&2; }
err()  { printf "[x] %s\n" "$*" 1>&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION=${2:-}; shift 2;;
    --zone) ZONE=${2:-}; shift 2;;
    -o|--out) OUT_FILE=${2:-}; shift 2;;
    --fetch-secrets) FETCH_SECRETS=1; shift;;
    --secret) SECRETS+=("${2:-}"); shift 2;;
    --ci-gsa) CI_GSA_EMAIL=${2:-}; shift 2;;
    --ci-cred-file) CI_CRED_FILE=${2:-}; shift 2;;
    --no-ci-detect) DETECT_CI_GSA=0; shift;;
    -h|--help) usage; exit 0;;
    *) err "Unknown argument: $1";;
  esac
done

command -v gcloud >/dev/null 2>&1 || err "gcloud not found in PATH. Install Google Cloud SDK."

# Auth
if ! gcloud auth list --format="value(account)" | grep -q .; then
  info "No gcloud account authenticated. Launching login..."
  gcloud auth login >/dev/null
else
  ACCTS=$(gcloud auth list --format="value(account)" | paste -sd, -)
  info "Found authenticated account(s): $ACCTS"
fi

# Project selection
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [[ -n "$CURRENT_PROJECT" && "$CURRENT_PROJECT" != "(unset)" ]]; then
  read -r -p "Keep current project [$CURRENT_PROJECT]? (Y/n) " keep
  keep=${keep:-Y}
  if [[ ${keep,,} == n ]]; then CURRENT_PROJECT=""; fi
fi

if [[ -z "$CURRENT_PROJECT" || "$CURRENT_PROJECT" == "(unset)" ]]; then
  info "Listing accessible projects..."
  mapfile -t PROJ_ROWS < <(gcloud projects list --format='csv[no-heading](projectId,name)')
  [[ ${#PROJ_ROWS[@]} -gt 0 ]] || err "No accessible GCP projects. Check permissions."
  echo
  echo "Select a GCP project:"
  i=1
  for row in "${PROJ_ROWS[@]}"; do
    IFS=',' read -r pid pname <<<"$row"
    echo "  [$i] $pid - $pname"
    i=$((i+1))
  done
  while :; do
    read -r -p "Enter number (1-${#PROJ_ROWS[@]}): " sel
    [[ "$sel" =~ ^[0-9]+$ ]] || { warn "Invalid input"; continue; }
    idx=$((sel-1))
    [[ $idx -ge 0 && $idx -lt ${#PROJ_ROWS[@]} ]] || { warn "Out of range"; continue; }
    IFS=',' read -r PROJECT _ <<<"${PROJ_ROWS[$idx]}"
    break
  done
else
  PROJECT=$CURRENT_PROJECT
fi

gcloud config set project "$PROJECT" >/dev/null
info "Project set to: $PROJECT"

# Region/Zone selection
if [[ -z "$REGION" ]]; then
  DEFAULT_REGION=$(gcloud config get-value run/region 2>/dev/null || true)
  if [[ -z "$DEFAULT_REGION" || "$DEFAULT_REGION" == "(unset)" ]]; then
    DEFAULT_REGION=$(gcloud config get-value compute/region 2>/dev/null || true)
  fi
  DEFAULT_REGION=${DEFAULT_REGION:-europe-west3}
  read -r -p "Preferred region [$DEFAULT_REGION]: " inp
  REGION=${inp:-$DEFAULT_REGION}
fi
gcloud config set compute/region "$REGION" >/dev/null

if [[ -z "$ZONE" ]]; then
  DEFAULT_ZONE=$(gcloud config get-value compute/zone 2>/dev/null || true)
  DEFAULT_ZONE=${DEFAULT_ZONE:-${REGION}-a}
  read -r -p "Preferred zone [$DEFAULT_ZONE]: " inpZ
  ZONE=${inpZ:-$DEFAULT_ZONE}
fi
gcloud config set compute/zone "$ZONE" >/dev/null
info "Region/Zone: $REGION / $ZONE"

# Detect CI/GitHub Actions Service Account (optional)
CI_GSA_ROLES=""
if [[ -z "$CI_GSA_EMAIL" ]]; then
  # Try to parse from provided credentials file
  if [[ -n "$CI_CRED_FILE" && -f "$CI_CRED_FILE" ]]; then
    if command -v jq >/dev/null 2>&1; then
      CI_GSA_EMAIL=$(jq -r '.client_email // empty' "$CI_CRED_FILE" || true)
    elif command -v python3 >/dev/null 2>&1; then
      CI_GSA_EMAIL=$(python3 - <<'PY'
import json,sys
try:
    print(json.load(open(sys.argv[1])).get('client_email',''))
except Exception:
    pass
PY
"$CI_CRED_FILE")
    else
      CI_GSA_EMAIL=$(grep -o '"client_email"\s*:\s*"[^"]\+"' "$CI_CRED_FILE" | head -1 | sed -E 's/.*:"([^"]+)"/\1/')
    fi
  fi
fi

if [[ -z "$CI_GSA_EMAIL" && $DETECT_CI_GSA -eq 1 ]]; then
  # Heuristic 1: list SAs and pick names with typical CI patterns
  mapfile -t SA_LIST < <(gcloud iam service-accounts list --format='value(email)')
  for sa in "${SA_LIST[@]}"; do
    if echo "$sa" | grep -Ei '(github|actions|ci|cd|deploy|pipeline)' >/dev/null; then
      CI_GSA_EMAIL="$sa"; break
    fi
  done

  # Heuristic 2: check IAM policy for SAs bound to deploy roles
  if [[ -z "$CI_GSA_EMAIL" ]]; then
    mapfile -t IAM_ROWS < <(gcloud projects get-iam-policy "$PROJECT" \
      --flatten="bindings[].members" \
      --filter="bindings.members:serviceAccount:" \
      --format='csv[no-heading](bindings.role,bindings.members)')
    # Collect candidate SAs by roles typically used in CI deployments
    wanted_roles='roles/run.admin|roles/run.developer|roles/artifactregistry.writer|roles/artifactregistry.admin|roles/cloudsql.client|roles/iam.serviceAccountUser|roles/container.admin|roles/container.developer'
    declare -A ROLE_BY_SA
    for row in "${IAM_ROWS[@]}"; do
      role=${row%%,*}
      member=${row#*,}
      case "$member" in serviceAccount:*) sa_email=${member#serviceAccount:};; *) continue;; esac
      if echo "$role" | grep -E "$wanted_roles" >/dev/null; then
        ROLE_BY_SA["$sa_email"]+="${role};"
      fi
    done
    # Prefer SAs matching CI patterns; otherwise the first with most roles
    best_sa=""; best_count=0
    for sa in "${!ROLE_BY_SA[@]}"; do
      count=$(printf '%s' "${ROLE_BY_SA[$sa]}" | tr ';' '\n' | grep -c . || true)
      if echo "$sa" | grep -Ei '(github|actions|ci|cd|deploy|pipeline)' >/dev/null; then
        CI_GSA_EMAIL="$sa"; CI_GSA_ROLES="${ROLE_BY_SA[$sa]}"; break
      fi
      if [[ $count -gt $best_count ]]; then
        best_count=$count; best_sa="$sa"; CI_GSA_ROLES="${ROLE_BY_SA[$sa]}"
      fi
    done
    if [[ -z "$CI_GSA_EMAIL" && -n "$best_sa" ]]; then CI_GSA_EMAIL="$best_sa"; fi
  fi
fi

# Redis (Memorystore)
REDIS_NAME=""; REDIS_HOST=""; REDIS_PORT=""
mapfile -t REDIS_ROWS < <(gcloud redis instances list --region "$REGION" --format='csv[no-heading](name,host,port)' 2>/dev/null || true)
if [[ ${#REDIS_ROWS[@]} -gt 0 ]]; then
  echo
  echo "Select Redis instance in $REGION (Memorystore):"
  i=1
  for row in "${REDIS_ROWS[@]}"; do
    IFS=',' read -r rn rh rp <<<"$row"; echo "  [$i] $rn - $rh:$rp"; i=$((i+1))
  done
  read -r -p "Enter number (1-${#REDIS_ROWS[@]}) or leave empty to skip: " sel
  if [[ -n "$sel" && "$sel" =~ ^[0-9]+$ ]]; then
    idx=$((sel-1)); if [[ $idx -ge 0 && $idx -lt ${#REDIS_ROWS[@]} ]]; then
      IFS=',' read -r REDIS_NAME REDIS_HOST REDIS_PORT <<<"${REDIS_ROWS[$idx]}"
    fi
  fi
else
  warn "No Redis instances found in region $REGION"
fi

# Cloud SQL (Postgres)
SQL_NAME=""; SQL_REGION=""; SQL_IP=""; SQL_CONN=""
mapfile -t SQL_ROWS < <(gcloud sql instances list --format='csv[no-heading](name,region,databaseVersion,ipAddresses[0].ipAddress,connectionName)' 2>/dev/null || true)
if [[ ${#SQL_ROWS[@]} -gt 0 ]]; then
  # Filter for POSTGRES
  FILTERED=()
  for row in "${SQL_ROWS[@]}"; do
    IFS=',' read -r sn sr sv sip sconn <<<"$row"
    [[ "$sv" == POSTGRES* ]] && FILTERED+=("$row")
  done
  if [[ ${#FILTERED[@]} -gt 0 ]]; then
    echo
    echo "Select Cloud SQL (Postgres) instance:"
    i=1
    for row in "${FILTERED[@]}"; do
      IFS=',' read -r sn sr sv sip sconn <<<"$row"
      echo "  [$i] $sn - $sr - IP: $sip"
      i=$((i+1))
    done
    read -r -p "Enter number (1-${#FILTERED[@]}) or leave empty to skip: " sel
    if [[ -n "$sel" && "$sel" =~ ^[0-9]+$ ]]; then
      idx=$((sel-1)); if [[ $idx -ge 0 && $idx -lt ${#FILTERED[@]} ]]; then
        IFS=',' read -r SQL_NAME SQL_REGION _ SQL_IP SQL_CONN <<<"${FILTERED[$idx]}"
      fi
    fi
  else
    warn "No Postgres Cloud SQL instances found"
  fi
else
  warn "No Cloud SQL instances found"
fi

# Cloud Run (managed)
mapfile -t RUN_ROWS < <(gcloud run services list --platform=managed --region "$REGION" --format='csv[no-heading](metadata.name,status.url)' 2>/dev/null || true)

# Artifact Registry (Docker)
mapfile -t AR_ROWS < <(gcloud artifacts repositories list --location "$REGION" --format='csv[no-heading](name,format,location)' 2>/dev/null || true)

# Write output
if [[ -e "$OUT_FILE" ]]; then
  read -r -p "$OUT_FILE exists. Overwrite? (y/N) " ans
  [[ ${ans,,} == y ]] || err "Aborted to avoid overwrite."
fi

umask 077
{
  echo "# Generated by scripts/gcloud-bootstrap.sh on $(date -Iseconds)"
  echo "# Do not commit this file. Contains environment hints from GCP."
  echo "GCP_PROJECT=$PROJECT"
  echo "GCP_REGION=$REGION"
  echo "GCP_ZONE=$ZONE"
  echo
  if [[ -n "$CI_GSA_EMAIL" ]]; then
    echo "# GitHub Actions / CI Service Account (detected or provided)"
    echo "GITHUB_ACTIONS_GSA_EMAIL=$CI_GSA_EMAIL"
    [[ -n "$CI_GSA_ROLES" ]] && echo "GITHUB_ACTIONS_GSA_ROLES=$(printf '%s' "$CI_GSA_ROLES" | sed 's/;$/\n/; s/;/,/g')"
    echo
  fi
  if [[ -n "$REDIS_HOST" ]]; then
    echo "# Redis (Memorystore)"
    echo "GCP_REDIS_INSTANCE=$REDIS_NAME"
    echo "GCP_REDIS_HOST=$REDIS_HOST"
    echo "GCP_REDIS_PORT=$REDIS_PORT"
    echo "GCP_REDIS_URL=redis://$REDIS_HOST:$REDIS_PORT/0"
    echo
  fi
  if [[ -n "$SQL_NAME" ]]; then
    echo "# Cloud SQL (Postgres)"
    echo "GCP_SQL_INSTANCE=$SQL_NAME"
    echo "GCP_SQL_REGION=$SQL_REGION"
    [[ -n "$SQL_IP" ]] && echo "GCP_SQL_PUBLIC_IP=$SQL_IP"
    [[ -n "$SQL_CONN" ]] && echo "GCP_SQL_CONNECTION_NAME=$SQL_CONN"
    echo "# For local dev, keep DB_USER/DB_PASSWORD from .env; optionally point DB_HOST to GCP_SQL_PUBLIC_IP"
    echo
  fi
  if [[ ${#RUN_ROWS[@]} -gt 0 ]]; then
    echo "# Cloud Run services in $REGION"
    for row in "${RUN_ROWS[@]}"; do
      IFS=',' read -r rn rurl <<<"$row"
      # Uppercase and sanitize name
      key=$(echo "$rn" | tr '[:lower:]' '[:upper:]' | sed -E 's/[^A-Z0-9]+/_/g')
      echo "GCP_RUN_${key}_URL=$rurl"
    done
    echo
  fi
  if [[ ${#AR_ROWS[@]} -gt 0 ]]; then
    echo "# Artifact Registry docker repos in $REGION"
    for row in "${AR_ROWS[@]}"; do
      IFS=',' read -r aname aformat aloc <<<"$row"
      [[ "$aformat" == "DOCKER" ]] || continue
      # Derive location from name if 'location' column is empty
      if [[ -z "$aloc" || "$aloc" == "(unset)" ]]; then
        # name pattern: projects/PROJECT/locations/LOCATION/repositories/REPO
        aloc=$(printf '%s' "$aname" | awk -F'/locations/' '{print $2}' | cut -d'/' -f1)
      fi
      # Fallback to selected REGION if still empty
      aloc=${aloc:-$REGION}
      host="${aloc}-docker.pkg.dev"
      repo="${aname##*/}"
      echo "GCP_ARTIFACT_REGISTRY_HOST=$host"
      echo "GCP_ARTIFACT_REGISTRY_REPO=$repo"
    done
    echo
  fi
} > "$OUT_FILE"

info "Wrote $OUT_FILE"

if [[ $FETCH_SECRETS -eq 1 ]]; then
  if [[ ${#SECRETS[@]} -eq 0 ]]; then
    warn "--fetch-secrets was set but no --secret provided. Skipping."
  else
    warn "Fetching secret payloads will write sensitive values to $OUT_FILE. Handle with care!"
    for s in "${SECRETS[@]}"; do
      if val=$(gcloud secrets versions access latest --secret="$s" 2>/dev/null); then
        val_escaped=$(printf '%s' "$val" | tr '\n' '\\n')
        printf '%s=%s\n' "$s" "$val_escaped" >> "$OUT_FILE"
        info "Wrote secret: $s"
      else
        warn "Failed to access secret: $s"
      fi
    done
  fi
fi

echo
echo "Done. Suggested next steps:"
echo "  - Review $OUT_FILE and selectively copy values into your .env as needed."
echo "  - Do not commit $OUT_FILE."
echo "  - Prefer local DB/Redis for dev; or use Cloud SQL Auth Proxy if needed."
