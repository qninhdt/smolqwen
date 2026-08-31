#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

key_path="$project_root/artifacts/serving/vllm-api-key"
tunnel_log="$project_root/artifacts/serving/cloudflared.log"
tunnel_pid="$project_root/artifacts/serving/cloudflared.pid"
mkdir -p "$(dirname "$key_path")"

if [[ ! -s "$key_path" ]]; then
  umask 077
  openssl rand -hex 32 > "$key_path"
fi
export VLLM_API_KEY="$(<"$key_path")"

# Compose is idempotent and owns the model/proxy lifetime. It exposes only the
# proxy port; vLLM remains on loopback in the shared network namespace.
docker compose up --detach --build vllm-server proxy

ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent --show-error \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    http://127.0.0.1:8080/v1/models >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "authenticated model readiness did not succeed" >&2
  exit 1
fi

command -v cloudflared >/dev/null 2>&1 || {
  echo "cloudflared is required after the authenticated proxy is ready" >&2
  exit 1
}

if [[ -s "$tunnel_pid" ]] && kill -0 "$(<"$tunnel_pid")" 2>/dev/null; then
  echo "reusing cloudflared pid $(<"$tunnel_pid")"
else
  : > "$tunnel_log"
  setsid cloudflared tunnel --url http://127.0.0.1:8080 \
    >"$tunnel_log" 2>&1 </dev/null &
  echo "$!" > "$tunnel_pid"
fi

base_url=""
for _ in $(seq 1 60); do
  base_url="$(sed -n 's#.*\(https://[-a-z0-9]*\.trycloudflare\.com\).*#\1#p' "$tunnel_log" | head -1)"
  [[ -n "$base_url" ]] && break
  sleep 1
done
if [[ -z "$base_url" ]]; then
  echo "cloudflared did not publish a URL; inspect $tunnel_log" >&2
  exit 1
fi

echo "base_url=$base_url"
echo "api_key_file=$key_path"
