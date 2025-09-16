#!/usr/bin/env bash
set -euo pipefail

# Liveliness (no auth required)
curl -s http://localhost:4000/health/liveliness | grep -qi "alive" && echo "LiteLLM alive"

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
