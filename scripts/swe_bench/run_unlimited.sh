#!/usr/bin/env bash
# Run the full SWE-bench Verified prediction suite with no per-instance wall clock.
# Avoids the common zsh multi-line trap where flags after a bare `runner.py` line
# are dropped and the runner silently uses the 600s built-in default.
#
# Usage:
#   bash scripts/swe_bench/run_unlimited.sh
#   bash scripts/swe_bench/run_unlimited.sh --instance-ids astropy__astropy-12907
#   bash scripts/swe_bench/run_unlimited.sh --limit 3 -v
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export MANA_SWE_BENCH_TIMEOUT="${MANA_SWE_BENCH_TIMEOUT:-0}"
exec python scripts/swe_bench/runner.py --timeout 0 --output predictions.jsonl "$@"
