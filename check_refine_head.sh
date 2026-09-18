#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

"${PYTHON}" tools/check_server_package.py

