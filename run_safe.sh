#!/usr/bin/env bash
# Run a command in its own systemd scope with a hard RAM cap, so if it runs out of
# memory only this job is killed instead of the whole PC freezing in swap.
#
#   ./run_safe.sh <name> <command...>
#   ./run_safe.sh pilot-bf16 .venv/bin/python generate.py ...
#
# Output goes to logs/<name>.log. Check with: tail -f logs/<name>.log
set -euo pipefail
cd "$(dirname "$0")"
name=$1; shift
mkdir -p logs
exec systemd-run --user --scope --quiet --unit="aq-$name-$(date +%s)" \
    -p MemoryMax="${MEM_MAX:-22G}" -p MemorySwapMax="${SWAP_MAX:-1G}" \
    "$@" > "logs/$name.log" 2>&1
