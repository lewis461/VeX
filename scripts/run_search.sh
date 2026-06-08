#!/bin/bash
# Thin wrapper around bin/search_index. Pass any search_index flags through.
# Example:
#   ./scripts/run_search.sh -index ./idx -base data.fvecs -query q.fvecs -nq 1000 -k 10 -nprobe 1
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/bin/search_index" "$@"
