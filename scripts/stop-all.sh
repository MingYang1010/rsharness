#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EO_HARNESS_ROOT:-/sata/yangm/eo-harness}"
DOCKER_CONTEXT="${EO_DOCKER_CONTEXT:-rootless}"

unset DOCKER_HOST

if docker --context "$DOCKER_CONTEXT" info >/dev/null 2>&1; then
  docker --context "$DOCKER_CONTEXT" compose \
    -f "$PROJECT_ROOT/compose.yaml" down --remove-orphans
fi

printf 'EO Harness containers stopped in Docker context: %s\n' \
  "$DOCKER_CONTEXT"
printf 'Rootless Docker data remains at /sata/yangm/docker-rootless/data.\n'
