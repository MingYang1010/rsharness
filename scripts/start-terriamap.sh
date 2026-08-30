#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EO_HARNESS_ROOT:-/sata/yangm/eo-harness}"

printf 'Starting the complete EO Harness stack.\n'
exec "$PROJECT_ROOT/scripts/start-harness.sh"
