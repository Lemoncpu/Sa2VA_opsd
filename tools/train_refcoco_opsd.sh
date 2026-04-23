#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "tools/train_refcoco_opsd.sh is kept as a compatibility wrapper and now forwards to the explicit 4B entry." >&2
exec bash "${ROOT_DIR}/tools/train_refcoco_opsd_4b.sh" "$@"
