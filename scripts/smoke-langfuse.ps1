Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Simple Langfuse/Django smoke checks executed from the host (PowerShell)
# - Verifies LANGFUSE_* env in containers
# - Checks Langfuse health from web container
# - Runs a minimal SDK smoke (observation) inside web
# - Hits /v1/ai/rag/query/ on the dev server to exercise graph + nodes

$compose = 'docker compose -f docker-compose.yml -f docker-compose.dev.yml'

function Invoke-InContainer {
    param(
        [Parameter(Mandatory)] [string] $Service,
        [Parameter(Mandatory)] [string] $Cmd
    )
    & $compose exec $Service sh -lc $Cmd
}

function Show-Env {
    param([string] $Service)
    Write-Host "[smoke] $Service environment (LANGFUSE_*)" -ForegroundColor Cyan
    Invoke-InContainer $Service 'echo $LANGFUSE_HOST && echo $LANGFUSE_PUBLIC_KEY && echo $LANGFUSE_SECRET_KEY && echo ${LANGFUSE_SAMPLE_RATE:-unset} && echo ${LANGFUSE_ENVIRONMENT:-unset}'
}

Write-Host '[smoke] Checking container environments' -ForegroundColor Yellow
Show-Env 'web'
Show-Env 'worker'
Show-Env 'ingestion-worker'

Write-Host '[smoke] Langfuse health from web container' -ForegroundColor Yellow
try {
    Invoke-InContainer 'web' 'curl -sS $LANGFUSE_HOST/api/public/health'
} catch {
    Write-Warning "Health check failed: $($_.Exception.Message)"
}

Write-Host '[smoke] Running SDK + Root Trace (v3/OTel) inside web container' -ForegroundColor Yellow
Invoke-InContainer 'web' @'
cat > /app/lf_smoke.py << "PY"
from opentelemetry.trace import get_tracer
from langfuse import observe, get_client

tr = get_tracer("noesis2.smoke")

@observe(name="smoke.child")
def child():
    return "ok"

with tr.start_as_current_span("smoke.trace"):
    print(child())

# Flush synchronously so the trace appears immediately in the UI
get_client().flush()
print("Flush complete.")
PY
OTEL_RESOURCE_ATTRIBUTES='service.name=noesis2,service.version=dev' python /app/lf_smoke.py
'@

Write-Host '[smoke] Running OTel-only exporter smoke inside web container' -ForegroundColor Yellow
Invoke-InContainer 'web' @'
cat > /app/lf_otel_only.py << "PY"
from opentelemetry.trace import get_tracer

tr = get_tracer("noesis2.otel-smoke")
with tr.start_as_current_span("smoke.otel.trace"):
    print("otel ok")
PY
OTEL_TRACES_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${LANGFUSE_HOST%/}/api/public/otel/v1/traces" \
OTEL_EXPORTER_OTLP_HEADERS="X-Langfuse-Public-Key=$LANGFUSE_PUBLIC_KEY,X-Langfuse-Secret-Key=$LANGFUSE_SECRET_KEY" \
OTEL_RESOURCE_ATTRIBUTES='service.name=noesis2,service.version=dev' \
python /app/lf_otel_only.py
'@

Write-Host '[smoke] Exercising POST /v1/ai/rag/query/ on dev server' -ForegroundColor Yellow
try {
    Invoke-InContainer 'web' @'
curl -sS -X POST http://localhost:8000/v1/ai/rag/query/ \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Schema: dev" \
  -H "X-Tenant-ID: dev" \
  -H "X-Case-ID: local" \
  --data '{"question":"Ping?"}' | head -c 200
'@
} catch {
    Write-Warning "RAG query failed: $($_.Exception.Message)"
}

Write-Host '[smoke] Done. Check Langfuse GUI:' -ForegroundColor Green
Write-Host " - Project: the project of your current keys" -ForegroundColor Green
Write-Host " - Environment: default (or All environments)" -ForegroundColor Green
Write-Host " - Data -> Observations: sdk.smoketest, node:compose, rag.hybrid.search" -ForegroundColor Green
Write-Host " - Traces: graph:<graph-name> (if root traces are enabled)" -ForegroundColor Green
