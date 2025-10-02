#!/usr/bin/env bash
set -euo pipefail

if [[ "${NO_COLOR:-}" == "1" ]]; then
  BOLD=""
  RESET=""
else
  BOLD="\033[1m"
  RESET="\033[0m"
fi

HOST=${NOESIS_HOST:-"http://demo.localhost:8000"}
TENANT_SCHEMA=${TENANT_SCHEMA:-"demo"}
TENANT_ID=${TENANT_ID:-"demo"}
CASE_ID=${CASE_ID:-"local"}
FILE_PATH=${RAG_DEMO_FILE:-"hello.txt"}
QUERY=${RAG_DEMO_QUERY:-"ZEBRAGURKE"}
ALPHA=${RAG_DEMO_ALPHA:-"0.6"}
MIN_SIM=${RAG_DEMO_MIN_SIM:-"0.15"}
VEC_LIMIT=${RAG_DEMO_VEC_LIMIT:-"20"}
LEX_LIMIT=${RAG_DEMO_LEX_LIMIT:-"30"}
TRGM_LIMIT=${RAG_DEMO_TRGM_LIMIT:-"0.30"}
TOP_K=${RAG_DEMO_TOP_K:-"5"}
MAX_POLLS=${RAG_DEMO_MAX_POLLS:-"12"}
SLEEP_SECONDS=${RAG_DEMO_POLL_INTERVAL:-"5"}
METADATA_JSON=${RAG_DEMO_METADATA:-"{\"external_id\":\"demo-hello\",\"label\":\"smoke\"}"}

usage() {
  cat <<USAGE
${BOLD}RAG Demo Walkthrough${RESET}

Dieses Skript automatisiert Upload → Ingestion → Hybrid-Suche gegen die lokale
Demo-Instanz. Voraussetzung: docker-compose Umgebung läuft, Worker bedient die
Queue \'ingestion\'.

Umgebung anpassen:
  NOESIS_HOST, TENANT_SCHEMA, TENANT_ID, CASE_ID
  RAG_DEMO_FILE, RAG_DEMO_METADATA, RAG_DEMO_QUERY
  RAG_DEMO_ALPHA, RAG_DEMO_MIN_SIM, RAG_DEMO_VEC_LIMIT, RAG_DEMO_LEX_LIMIT,
  RAG_DEMO_TRGM_LIMIT, RAG_DEMO_TOP_K, RAG_DEMO_MAX_POLLS, RAG_DEMO_POLL_INTERVAL

Beispiel:
  TENANT_SCHEMA=demo TENANT_ID=demo CASE_ID=local \\
    ./scripts/rag_demo_walkthrough.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for cmd in curl jq mktemp; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[rag-demo] Fehler: benötigtes Programm '$cmd' fehlt." >&2
    exit 1
  fi
fi

if [[ ! -f "$FILE_PATH" ]]; then
  echo "[rag-demo] Hinweis: '$FILE_PATH' existiert nicht – Beispielinhalt wird erstellt."
  cat <<'TXT' > "$FILE_PATH"
Hallo ZEBRAGURKE,
dies ist ein End-to-End-Test für die RAG-Demo.
TXT
fi

if [[ ! -s "$FILE_PATH" ]]; then
  echo "[rag-demo] Fehler: '$FILE_PATH' ist leer." >&2
  exit 1
fi

TMP_METADATA=$(mktemp)
printf '%s' "$METADATA_JSON" > "$TMP_METADATA"

printf '%b[1/4] Upload %s%b\n' "$BOLD" "$FILE_PATH" "$RESET"
UPLOAD_JSON=$(curl --fail-with-body -sS -X POST "$HOST/ai/rag/documents/upload/" \
  -H "X-Tenant-Schema: $TENANT_SCHEMA" \
  -H "X-Tenant-Id: $TENANT_ID" \
  -H "X-Case-Id: $CASE_ID" \
  -F "file=@$FILE_PATH" \
  -F "metadata=@$TMP_METADATA;type=application/json")

document_id=$(jq -r '.document_id // empty' <<<"$UPLOAD_JSON")
external_id=$(jq -r '.external_id // empty' <<<"$UPLOAD_JSON")
if [[ -z "$document_id" ]]; then
  echo "[rag-demo] Fehler: Upload-Antwort enthält keine document_id." >&2
  echo "$UPLOAD_JSON" | jq . >&2
  exit 1
fi

printf '%bUpload Response:%b\n%s\n' "$BOLD" "$RESET" "$(jq . <<<"$UPLOAD_JSON")"

TMP_PAYLOAD=$(mktemp)
jq --arg doc "$document_id" '{document_ids:[$doc], priority:"normal"}' > "$TMP_PAYLOAD"

printf '%b[2/4] Ingestion-Run für %s%b\n' "$BOLD" "$document_id" "$RESET"
INGESTION_JSON=$(curl --fail-with-body -sS -X POST "$HOST/ai/rag/ingestion/run/" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Schema: $TENANT_SCHEMA" \
  -H "X-Tenant-Id: $TENANT_ID" \
  -H "X-Case-Id: $CASE_ID" \
  -d @"$TMP_PAYLOAD")

printf '%bIngestion Response:%b\n%s\n' "$BOLD" "$RESET" "$(jq . <<<"$INGESTION_JSON")"
invalid_ids=$(jq -r '.invalid_ids | @csv' <<<"$INGESTION_JSON" 2>/dev/null || true)
if [[ -n "${invalid_ids:-}" && "${invalid_ids}" != "null" ]]; then
  echo "[rag-demo] Warnung: Ungültige Dokument-IDs -> ${invalid_ids}" >&2
fi

printf '%b[3/4] Warte auf RAG-Treffer für external_id=%s%b\n' "$BOLD" "${external_id:-<unbekannt>}" "$RESET"

TMP_SEARCH=$(mktemp)
SUCCESS=0
for ((i=1; i<=MAX_POLLS; i++)); do
  jq -n \
    --arg query "$QUERY" \
    --argjson top_k "$TOP_K" \
    --argjson alpha "$ALPHA" \
    --argjson min_sim "$MIN_SIM" \
    --argjson vec_limit "$VEC_LIMIT" \
    --argjson lex_limit "$LEX_LIMIT" \
    --argjson trgm_limit "$TRGM_LIMIT" \
    '{query:$query, top_k:$top_k, alpha:$alpha, min_sim:$min_sim, vec_limit:$vec_limit, lex_limit:$lex_limit, trgm_limit:$trgm_limit}' \
    > "$TMP_SEARCH"

  RESPONSE=$(curl --fail-with-body -sS -X POST "$HOST/ai/v1/rag-demo/" \
    -H "Content-Type: application/json" \
    -H "X-Tenant-Schema: $TENANT_SCHEMA" \
    -H "X-Tenant-Id: $TENANT_ID" \
    -H "X-Case-Id: $CASE_ID" \
    -d @"$TMP_SEARCH")

  error_field=$(jq -r '.error // empty' <<<"$RESPONSE")
  matches_with_external=$(jq --arg ext "$external_id" '(.matches // []) | map(select((.metadata.external_id // "") == $ext)) | length' <<<"$RESPONSE")

  printf '[Versuch %d/%d] matches=%s error=%s\n' "$i" "$MAX_POLLS" "$matches_with_external" "${error_field:-<none>}"

  if [[ -z "$error_field" && "$matches_with_external" -gt 0 ]]; then
    SUCCESS=1
    FINAL_RESPONSE="$RESPONSE"
    break
  fi

  sleep "$SLEEP_SECONDS"
done

if [[ "$SUCCESS" -ne 1 ]]; then
  echo "[rag-demo] Fehler: Kein Treffer mit externer ID innerhalb des Zeitfensters gefunden." >&2
  exit 1
fi

printf '%b[4/4] Hybrid-Suche erfolgreich%b\n' "$BOLD" "$RESET"
printf '%bFinale Antwort:%b\n%s\n' "$BOLD" "$RESET" "$(jq . <<<"$FINAL_RESPONSE")"

rm -f "$TMP_METADATA" "$TMP_PAYLOAD" "$TMP_SEARCH"
exit 0
