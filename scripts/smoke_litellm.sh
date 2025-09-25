#!/usr/bin/env bash
# Be tolerant if pipefail isn't supported in some shells on Windows
set -euo
{ set -o pipefail; } 2>/dev/null || true

# Liveliness (no auth required) with small retry loop
for i in $(seq 1 10); do
  if curl -s http://localhost:4000/health/liveliness | grep -qi "alive"; then
    echo "LiteLLM alive"; break
  fi
  sleep 1
done

# Optional: readiness (auth required)
curl -s -H "Authorization: Bearer ${LITELLM_MASTER_KEY:-}" http://localhost:4000/health | grep -qi '"unhealthy_count":0' && echo "LiteLLM healthy"

cat > /tmp/litellm_chat.json << 'JSON'
{
  "model": "gemini-2.5-flash",
  "messages": [{"role": "user", "content": "Sag \"ok\""}]
}
JSON

curl -s -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY:-}" \
  --data-binary @/tmp/litellm_chat.json | grep -qi '"choices"' && echo "Chat OK"
