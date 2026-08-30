#!/usr/bin/env bash
set -euo pipefail

DOCKER_CONTEXT="${EO_DOCKER_CONTEXT:-rootless}"
EXPECTED_DATA_ROOT="${EO_DOCKER_DATA_ROOT:-/sata/yangm/docker-rootless/data}"

unset DOCKER_HOST

if ! systemctl --user is-active --quiet docker.service; then
  systemctl --user start docker.service
fi

for attempt in $(seq 1 30); do
  if docker --context "$DOCKER_CONTEXT" info >/dev/null 2>&1; then
    actual_data_root="$(
      docker --context "$DOCKER_CONTEXT" info --format '{{.DockerRootDir}}'
    )"

    if [ "$actual_data_root" != "$EXPECTED_DATA_ROOT" ]; then
      printf 'Unexpected Docker data root: %s\n' "$actual_data_root" >&2
      printf 'Expected SATA data root: %s\n' "$EXPECTED_DATA_ROOT" >&2
      exit 1
    fi

    printf 'Rootless Docker is ready: context=%s\n' "$DOCKER_CONTEXT"
    printf 'Docker data root: %s\n' "$actual_data_root"
    exit 0
  fi
  sleep 1
done

systemctl --user status docker.service --no-pager >&2 || true
printf 'Rootless Docker did not become ready.\n' >&2
exit 1
