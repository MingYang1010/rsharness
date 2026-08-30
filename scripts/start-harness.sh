#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EO_HARNESS_ROOT:-/sata/yangm/eo-harness}"
DOCKER_CONTEXT="${EO_DOCKER_CONTEXT:-rootless}"
TERRIA_PORT="${TERRIA_PORT:-3001}"
HARNESS_API_PORT="${HARNESS_API_PORT:-8000}"
TERRIA_IMAGE="eo-harness/terriamap:0.4.6"
HARNESS_API_IMAGE="eo-harness/harness-api:0.2.0"
TERRIA_SOURCE_IMAGE="ghcr.io/terriajs/terriamap:0.4.6@sha256:0853ec153c53cef6ae926c99698ccc5cf4cc1a89906550068d3a36d95c3d54e0"

"$PROJECT_ROOT/scripts/start-project-docker.sh"
unset DOCKER_HOST

mkdir -p "$PROJECT_ROOT/state"

if ! docker --context "$DOCKER_CONTEXT" image inspect \
  "$TERRIA_IMAGE" >/dev/null 2>&1; then
  docker --context "$DOCKER_CONTEXT" pull "$TERRIA_SOURCE_IMAGE"
  docker --context "$DOCKER_CONTEXT" tag \
    "$TERRIA_SOURCE_IMAGE" "$TERRIA_IMAGE"
fi

if ! docker --context "$DOCKER_CONTEXT" image inspect \
  "$HARNESS_API_IMAGE" >/dev/null 2>&1; then
  docker --context "$DOCKER_CONTEXT" compose \
    -f "$PROJECT_ROOT/compose.yaml" build harness-api
fi
docker --context "$DOCKER_CONTEXT" compose \
  -f "$PROJECT_ROOT/compose.yaml" up -d
docker --context "$DOCKER_CONTEXT" compose \
  -f "$PROJECT_ROOT/compose.yaml" ps

for attempt in $(seq 1 60); do
  terria_ready=false
  api_ready=false

  if curl --noproxy '*' --fail --silent --show-error \
    "http://127.0.0.1:${TERRIA_PORT}/" >/dev/null 2>&1; then
    terria_ready=true
  fi
  if curl --noproxy '*' --fail --silent --show-error \
    "http://127.0.0.1:${HARNESS_API_PORT}/healthz" >/dev/null 2>&1; then
    api_ready=true
  fi

  if [[ "$terria_ready" == true && "$api_ready" == true ]]; then
    printf 'TerriaMap is ready: http://127.0.0.1:%s/\n' "$TERRIA_PORT"
    printf 'Harness API is ready: http://127.0.0.1:%s/docs\n' \
      "$HARNESS_API_PORT"
    exit 0
  fi
  sleep 2
done

docker --context "$DOCKER_CONTEXT" compose \
  -f "$PROJECT_ROOT/compose.yaml" logs --tail 100 >&2 || true
printf 'EO Harness stack did not become HTTP-ready.\n' >&2
exit 1
