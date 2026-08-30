#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EO_HARNESS_SOURCE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT_ROOT"
"$PYTHON" -m unittest discover \
  -s harness_api/tests \
  -p 'test_*.py' \
  -v
