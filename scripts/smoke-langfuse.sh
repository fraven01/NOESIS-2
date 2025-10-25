#!/usr/bin/env bash
set -euo pipefail

# Simple Langfuse/Django smoke checks executed from the host (bash)
# - Verifies LANGFUSE_* env in containers
# - Checks Langfuse health from web container
# - Runs a minimal SDK smoke (observation) inside web
# - Hits /ai/scope/ on the dev server to exercise graph + nodes

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.dev.yml)

in_container() {
  local service="$1"; shift
  "${COMPOSE[@]}" exec -T "$service" sh -lc "$*"
}

show_env() {
  local service="$1"
  echo "[smoke] $service environment (LANGFUSE_*)"
  in_container "$service" 'echo $LANGFUSE_HOST; echo $LANGFUSE_PUBLIC_KEY; echo $LANGFUSE_SECRET_KEY; echo ${LANGFUSE_SAMPLE_RATE:-unset}; echo ${LANGFUSE_ENVIRONMENT:-unset}'
}

echo "[smoke] Checking container environments"
show_env web
show_env worker
show_env ingestion-worker

LANGFUSE_HOST_VALUE="$(in_container web 'printf %s "${LANGFUSE_HOST:-}"')"
if [ -z "$LANGFUSE_HOST_VALUE" ]; then
  echo "[smoke] WARNING: LANGFUSE_HOST missing in web container env; falling back to http://langfuse:3000"
  LANGFUSE_HOST_VALUE="http://langfuse:3000"
fi
LANGFUSE_PUBLIC_KEY_VALUE="$(in_container web 'printf %s "${LANGFUSE_PUBLIC_KEY:-}"')"
LANGFUSE_SECRET_KEY_VALUE="$(in_container web 'printf %s "${LANGFUSE_SECRET_KEY:-}"')"
LANGFUSE_SAMPLE_RATE_VALUE="$(in_container web 'printf %s "${LANGFUSE_SAMPLE_RATE:-}"')"
LANGFUSE_ENVIRONMENT_VALUE="$(in_container web 'printf %s "${LANGFUSE_ENVIRONMENT:-}"')"
LANGFUSE_HOST_VALUE_STRIPPED="${LANGFUSE_HOST_VALUE%/}"

echo "[smoke] Langfuse health from web container"
in_container web 'curl -sS $LANGFUSE_HOST/api/public/health || true'

echo "[smoke] Running SDK + Root Trace (v3/OTel) inside web container"
in_container web "cat > /app/lf_smoke.py << 'PY'
from opentelemetry.trace import get_tracer
from langfuse import observe, get_client

tr = get_tracer('noesis2.smoke')

@observe(name='smoke.child')
def child():
    return 'ok'

with tr.start_as_current_span('smoke.trace'):
    print(child())

# Flush synchronously so the trace appears immediately in the UI
get_client().flush()
print('Flush complete.')
PY
OTEL_RESOURCE_ATTRIBUTES="service.name=noesis2,service.version=dev" python /app/lf_smoke.py"

echo "[smoke] Running OTel-only exporter smoke inside web container"
in_container web "cat > /app/lf_otel_only.py << 'PY'
from opentelemetry.trace import get_tracer

tr = get_tracer('noesis2.otel-smoke')
with tr.start_as_current_span('smoke.otel.trace'):
    print('otel ok')
PY
OTEL_TRACES_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT='${LANGFUSE_HOST_VALUE_STRIPPED}/api/public/otel/v1/traces' \
OTEL_EXPORTER_OTLP_HEADERS='X-Langfuse-Public-Key=${LANGFUSE_PUBLIC_KEY_VALUE},X-Langfuse-Secret-Key=${LANGFUSE_SECRET_KEY_VALUE}' \
OTEL_RESOURCE_ATTRIBUTES='service.name=noesis2,service.version=dev' \
python /app/lf_otel_only.py"

echo "[smoke] Exercising POST /ai/scope/ on dev server"
in_container web "curl -sS -X POST http://localhost:8000/ai/scope/ \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Schema: dev' \
  -H 'X-Tenant-ID: dev' \
  -H 'X-Case-ID: local' \
  --data '{}' | head -c 200 || true"

echo "[smoke] Done. Check Langfuse GUI:"
echo " - Project: the project of your current keys"
echo " - Environment: default (or All environments)"
echo " - Data -> Observations: sdk.smoketest, node:compose, rag.hybrid.search"
echo " - Traces: graph:<graph-name> (if root traces are enabled)"
