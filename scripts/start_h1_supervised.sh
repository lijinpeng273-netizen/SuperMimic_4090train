#!/usr/bin/env bash
# Run in a visible terminal; the supervisor survives closing the viewing window.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
resume_args=()
hours=3
session_name="h1_supervised_all_$(date +%Y%m%d_%H%M%S)"
while (( $# )); do
  case "$1" in
    --hours) hours="${2:?Missing hours}"; shift 2 ;;
    --continue-session) session_name="$(basename -- "${2:?Missing session}")"; resume_args=(--continue-session); shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
console="$project_dir/logs/$session_name.console.log"
mkdir -p logs
printf 'Starting %s-hour H1 supervision; quality early-stop disabled.\nReport: %s/docs/%s.md\nConsole: %s\n' "$hours" "$project_dir" "$session_name" "$console"
nohup ./scripts/local.sh python -B -u scripts/supervise_h1.py supervise \
  --session "$project_dir/logs/$session_name" --hours "$hours" "${resume_args[@]}" \
  >> "$console" 2>&1 < /dev/null &
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" > "$project_dir/logs/$session_name.pid"
printf 'Supervisor PID: %s\nStop command: kill -TERM %s\n\n' "$supervisor_pid" "$supervisor_pid"
tail --pid="$supervisor_pid" --sleep-interval=1 -n +1 -F "$console"
wait "$supervisor_pid"
