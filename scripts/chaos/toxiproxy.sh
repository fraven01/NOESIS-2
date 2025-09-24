#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
COMPOSE_CMD="${COMPOSE_CMD:-docker compose}"
TOXIPROXY_SERVICE="${TOXIPROXY_SERVICE:-toxiproxy}"
DB_PROXY_NAME="${TOXIPROXY_DB_PROXY:-postgres_chaos}"
REDIS_PROXY_NAME="${TOXIPROXY_REDIS_PROXY:-redis_chaos}"
LATENCY_MS="${CHAOS_LATENCY_MS:-250}"
JITTER_MS="${CHAOS_JITTER_MS:-75}"
BANDWIDTH_KBPS="${CHAOS_BANDWIDTH_KBPS:-256}"
DROP_TIMEOUT_MS="${CHAOS_DROP_TIMEOUT_MS:-1500}"
DROP_TOXICITY="${CHAOS_DROP_TOXICITY:-0.2}"
CHAOS_LOG_DIR="${CHAOS_LOG_DIR:-logs/chaos}"
CHAOS_LOG_FILE="$CHAOS_LOG_DIR/toxiproxy.log"
TOXIC_LATENCY_NAME="latency_downstream"
TOXIC_BANDWIDTH_NAME="bandwidth_downstream"
TOXIC_DROP_NAME="reset_peer_downstream"

log_event() {
  mkdir -p "$CHAOS_LOG_DIR"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf "%s %s\n" "$timestamp" "$1" | tee -a "$CHAOS_LOG_FILE" >&2
}

exec_cli() {
  $COMPOSE_CMD exec -T "$TOXIPROXY_SERVICE" toxiproxy-cli "$@"
}

ensure_service() {
  $COMPOSE_CMD up -d "$TOXIPROXY_SERVICE" >/dev/null
}

add_toxic() {
  local proxy_name=$1
  local toxic_name=$2
  local toxic_type=$3
  shift 3
  exec_cli toxic remove -n "$toxic_name" "$proxy_name" >/dev/null 2>&1 || true
  exec_cli toxic add "$proxy_name" -n "$toxic_name" -t "$toxic_type" --downstream "$@"
}

remove_toxic() {
  local proxy_name=$1
  local toxic_name=$2
  exec_cli toxic remove -n "$toxic_name" "$proxy_name" >/dev/null 2>&1 || true
}

apply_slow_net() {
  ensure_service
  add_toxic "$DB_PROXY_NAME" "$TOXIC_LATENCY_NAME" latency -a latency="$LATENCY_MS" -a jitter="$JITTER_MS"
  add_toxic "$DB_PROXY_NAME" "$TOXIC_BANDWIDTH_NAME" bandwidth -a rate="$BANDWIDTH_KBPS"
  add_toxic "$DB_PROXY_NAME" "$TOXIC_DROP_NAME" reset_peer -a timeout="$DROP_TIMEOUT_MS" -a toxicity="$DROP_TOXICITY"

  add_toxic "$REDIS_PROXY_NAME" "$TOXIC_LATENCY_NAME" latency -a latency="$LATENCY_MS" -a jitter="$JITTER_MS"
  add_toxic "$REDIS_PROXY_NAME" "$TOXIC_BANDWIDTH_NAME" bandwidth -a rate="$BANDWIDTH_KBPS"
  add_toxic "$REDIS_PROXY_NAME" "$TOXIC_DROP_NAME" reset_peer -a timeout="$DROP_TIMEOUT_MS" -a toxicity="$DROP_TOXICITY"

  log_event "SLOW_NET=enabled latency=${LATENCY_MS}ms jitter=${JITTER_MS}ms bandwidth=${BANDWIDTH_KBPS}KB/s drop_timeout=${DROP_TIMEOUT_MS}ms drop_toxicity=${DROP_TOXICITY}"
}

clear_toxics() {
  ensure_service
  for proxy in "$DB_PROXY_NAME" "$REDIS_PROXY_NAME"; do
    remove_toxic "$proxy" "$TOXIC_LATENCY_NAME"
    remove_toxic "$proxy" "$TOXIC_BANDWIDTH_NAME"
    remove_toxic "$proxy" "$TOXIC_DROP_NAME"
  done
  log_event "SLOW_NET=disabled"
}

show_status() {
  ensure_service
  exec_cli list
  printf "\nProxy inspection (%s):\n" "$DB_PROXY_NAME"
  exec_cli inspect "$DB_PROXY_NAME"
  printf "\nProxy inspection (%s):\n" "$REDIS_PROXY_NAME"
  exec_cli inspect "$REDIS_PROXY_NAME"
}

case "$ACTION" in
  enable)
    if [[ "${SLOW_NET:-false}" != "true" ]]; then
      echo "Refusing to enable toxics because SLOW_NET=true is required" >&2
      exit 1
    fi
    apply_slow_net
    ;;
  disable)
    clear_toxics
    ;;
  status)
    show_status
    ;;
  *)
    cat >&2 <<'USAGE'
Usage: scripts/chaos/toxiproxy.sh [enable|disable|status]
Environment:
  SLOW_NET=true              Required for enable to guard against accidental runs
  CHAOS_LATENCY_MS           Override downstream latency in milliseconds (default: 250)
  CHAOS_JITTER_MS            Override latency jitter in milliseconds (default: 75)
  CHAOS_BANDWIDTH_KBPS       Maximum downstream bandwidth in KB/s (default: 256)
  CHAOS_DROP_TIMEOUT_MS      Timeout before reset_peer triggers in milliseconds (default: 1500)
  CHAOS_DROP_TOXICITY        Percentage of connections impacted (0.0-1.0, default: 0.2)
USAGE
    exit 1
    ;;
esac
