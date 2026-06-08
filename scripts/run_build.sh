#!/bin/bash
# Thin wrapper around bin/build_index. Pass any build_index flags through.
# Example:
#   ./scripts/run_build.sh -base data.fvecs -output ./idx -merge pca -nested on -tau 1.01 -clusters 1000
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/bin/build_index" "$@"
